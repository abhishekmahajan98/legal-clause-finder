import concurrent.futures
from backend.utils import azure_utils as azure
from azure.search.documents.models import VectorizedQuery
import numpy as np
import json, tiktoken
from math import ceil
import logging
import sys
import traceback
import time  # Added for implementing backoff
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
class LLMPipelineError(Exception):
    """Custom exception for LLM pipeline errors."""
    pass

class LLMQueryPipeline:
    def __init__(self, max_retries=3, backoff_factor=2):
        """
        Initialize the LLMQueryPipeline.

        :param max_retries: Maximum number of retry attempts for failed validations.
        :param backoff_factor: Factor by which the wait time increases after each retry.
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def retrieve_document_chunks(self, document_id, max_results=10000):
        """
        Fetch all chunks of a single document using pagination.
        """
        try:
            search_client = azure.get_ai_search_client()
            document_filters = f"document_id eq '{document_id.upper()}'"

            batch_size = 100
            total_batches = ceil(max_results / batch_size)
            all_chunks = []
            skip = 0

            for batch in range(total_batches):
                current_top = min(batch_size, max_results - len(all_chunks))
                results = search_client.search(
                    search_text="*",  # Assuming you want all content; adjust as needed
                    top=current_top,
                    skip=skip,
                    select=[
                        "account",
                        "client_name",
                        "page_number",
                        "document_category",
                        "document_title",
                        "link",
                        "content",
                        "document_id"
                    ],
                    filter=document_filters,
                    semantic_configuration_name="my-semantic-config",
                    query_type="simple",
                    search_mode="any",
                )

                batch_chunks = [result for result in results]
                if not batch_chunks:
                    break  # No more results
                all_chunks.extend(batch_chunks)
                skip += batch_size

                if len(batch_chunks) < batch_size:
                    break  # No more results

            return all_chunks
        except Exception as e:
            logging.error("Error in retrieve_document_chunks: %s", str(e))
            logging.debug(traceback.format_exc())
            raise LLMPipelineError(f"Failed to retrieve chunks for document ID {document_id}.") from e

    def assemble_context(self, chunks):
        """
        Convert fetched chunks into context format for the prompt.
        """
        try:
            context = []
            for chunk in chunks:
                context.append({
                    "Account": chunk.get("account", "N/A"),
                    "Client Name": chunk.get("client_name", "N/A"),
                    "Page Number": chunk.get("page_number", "N/A"),
                    "Document Category": chunk.get("document_category", "N/A"),
                    "Document Title": chunk.get("document_title", "N/A"),
                    "Link": chunk.get("link", "#"),
                    "Content": chunk.get("content", ""),
                })
            return context
        except Exception as e:
            logging.error("Error in assemble_context: %s", str(e))
            logging.debug(traceback.format_exc())
            raise LLMPipelineError("Failed to assemble context.") from e

    def split_context(self, context, max_tokens=100000):
        """
        Split the context into smaller chunks each within the max_tokens limit.
        """
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            current_chunk = []
            current_tokens = 0
            all_chunks = []

            for entry in context:
                entry_str = json.dumps(entry, indent=2)
                entry_tokens = len(encoding.encode(entry_str))

                if current_tokens + entry_tokens > max_tokens:
                    if current_chunk:
                        all_chunks.append(current_chunk)
                        current_chunk = []
                        current_tokens = 0

                current_chunk.append(entry)
                current_tokens += entry_tokens

            if current_chunk:
                all_chunks.append(current_chunk)

            return all_chunks
        except Exception as e:
            logging.error("Error in split_context: %s", str(e))
            logging.debug(traceback.format_exc())
            raise LLMPipelineError("Failed to split context into chunks.") from e

    def map_function(self, context_chunk, query, conversation_history):
        """
        Generates a response for a context chunk and user query.
        """
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            context_str = json.dumps(context_chunk, indent=2)
            context_tokens = len(encoding.encode(context_str))

            if context_tokens > 100000:
                logging.error("Context chunk exceeds the maximum token limit.")
                raise LLMPipelineError("Context chunk is over 100000 tokens.")

            system_message = {
                "role": "system",
                "content": '''You are an advanced AI assistant specialized in supporting
                    the legal team with contract analysis. Your primary function
                    is to help identify, extract, and summarize specific clauses or language
                    within various types of contracts. Ensure all responses strictly adhere to the provided guidelines and formats.''',
            }

            context_message = {
                "role": "user",
                "content": f"Context (JSON format):\n{context_str}"
            }

            history_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in conversation_history
            ]

            user_query_message = {
                "role": "user",
                "content": f"User's question: {query}"
            }

            prompt_instructions = {
                "role": "user",
                "content": """Provide a concise answer based on the given context and conversation history.
                The context is from a subset of a document. If there are no matches, simply return 'No matches found for the query' exactly.
                If there are match(es):
                - Always mention the page number the information comes from. Also identify the section of the document the citation is under and mention it in the response
                - Cite the actual words from the document as well. Make sure there is enough context around the match in the citation
                - Give a brief summary of the section the citation is from
                - If a citation spans across multiple pages then always mention the page number as the lowest page where the citation starts from
                - Use the following as an example output to ensure the formatting closely matches exactly like the example. Do not deviate from this format in any way:

                Example Output:

                1. **Page: page_number**
                    - Under Section : Section Number and Section Heading
                    - Section Summary: "summary of the section the citation is derived from"
                    - Cited Text: "content to be cited"

                2. **Page: page_number**
                    - Under Section : Section Number and Section Heading
                    - Section Summary: "summary of the section the citation is derived from"
                    - Cited Text: "content to be cited"

                Only provide the result in the given format. Do not hallucinate or use information that is not provided in the prompt."""
            }

            messages = [system_message, context_message] + history_messages + [user_query_message, prompt_instructions]

            openai_client, deployment_name = azure.get_gpt4o_client()

            response = openai_client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                temperature=0.3,  # low temp for more deterministic output
                stop=None,  # Define stop sequences if needed
            )

            response_content = response.choices[0].message.content.strip()

            return response_content
        except LLMPipelineError:
            raise
        except Exception as e:
            logging.error("Error in map_function: %s", str(e))
            logging.debug(traceback.format_exc())
            raise LLMPipelineError("Failed to generate LLM response for the chunk.") from e

    def reduce_function(self, partial_responses, max_tokens=10000):
        """
        Summarize the list of partial responses into a final coherent response.
        If the combined responses exceed max_tokens, perform recursive summarization.
        """
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            combined_responses = "\n\n<subresponse>".join(partial_responses)
            combined_tokens = len(encoding.encode(combined_responses))

            if combined_tokens <= max_tokens:
                # Proceed to summarize
                return self._generate_summary(combined_responses)
            else:
                # Split into smaller batches and summarize each batch first
                batch_size = 10  # Adjust based on average response length to stay within limits
                batched_responses = [partial_responses[i:i + batch_size] for i in range(0, len(partial_responses), batch_size)]
                intermediate_summaries = []
                for batch in batched_responses:
                    intermediate_combined = "\n\n<subresponse>".join(batch)
                    intermediate_summaries.append(self._generate_summary(intermediate_combined))
                # Recursively summarize the intermediate summaries
                return self.reduce_function(intermediate_summaries, max_tokens)
        except Exception as e:
            logging.error("Error in reduce_function: %s", str(e))
            logging.debug(traceback.format_exc())
            raise LLMPipelineError("Failed to reduce partial responses.") from e

    def _generate_summary(self, combined_responses):
        """
        Helper method to generate a summary from combined responses.
        """
        try:
            system_message = {
                "role": "system",
                "content": '''You are an advanced AI assistant specialized in summarizing information.
                        Your task is to succinctly combine multiple summaries into a single coherent summary.
                        Ensure strict adherence to the provided format and avoid any hallucinations.''',
            }

            user_message = {
                "role": "user",
                "content": f"""
                Given the following summaries, generate a single concise and coherent summary. Each subquery response is separated by <subresponse> tags.
                - Ignore any subquery that contains phrases like "No matches from this subquery" or similar.
                - If all subqueries indicate no matches, respond with: "No matches found for the query.".
                - Ensure the final summary strictly follows the specified format without deviations.

                Format:

                1. **Page: page_number**
                    - Under Section : Section Number and Section Heading
                    - Section Summary: "summary of the section the citation is derived from"
                    - Cited Text: "content to be cited"

                2. **Page: page_number**
                    - Under Section : Section Number and Section Heading
                    - Section Summary: "summary of the section the citation is derived from"
                    - Cited Text: "content to be cited"

                Summaries:
                {combined_responses}
                """
            }

            openai_client, deployment_name = azure.get_gpt4o_client()

            response = openai_client.chat.completions.create(
                model=deployment_name,
                messages=[system_message, user_message],
                temperature=0,  # Set temperature to 0 for deterministic output
                stop=None,  # Define stop sequences if needed
            )

            summary = response.choices[0].message.content.strip()

            return summary
        except Exception as e:
            logging.error("Error in _generate_summary: %s", str(e))
            logging.debug(traceback.format_exc())
            raise LLMPipelineError("Failed to generate summary.") from e

    def process_query(self, query, document_id, conversation_history):
        """Processes the query for a single document_id."""
        try:
            if not document_id:
                logging.info("No document ID provided.")
                return "No matches found for the query"

            # Fetch all chunks of the specified document
            chunks = self.retrieve_document_chunks(document_id, max_results=10000)  # Adjust as needed
            if not chunks:
                logging.info(f"No chunks retrieved for document ID: {document_id}")
                return "No matches found for the query"

            # Assemble context from chunks
            context = self.assemble_context(chunks)
            logging.info(f"Number of context entries: {len(context)}")

            # Split context into manageable chunks for processing
            context_chunks = self.split_context(context, max_tokens=100000)  # Adjust max_tokens as needed
            logging.info("Number of context chunks: %d", len(context_chunks))

            partial_responses = []

            # Define the number of worker threads; adjust as needed
            max_workers = min(10, len(context_chunks))  # For example, up to 10 threads

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Prepare the tasks
                future_to_chunk = {
                    executor.submit(
                        self.map_function, chunk, query, conversation_history
                    ): idx for idx, chunk in enumerate(context_chunks)
                }

                for future in concurrent.futures.as_completed(future_to_chunk):
                    idx = future_to_chunk[future]
                    try:
                        partial_response = future.result()
                        if partial_response:  # Ensure there is a response to summarize
                            partial_responses.append(partial_response)
                            logging.debug(f"Partial response {idx + 1} added.")
                    except LLMPipelineError as e:
                        logging.error("Error processing chunk %d: %s", idx + 1, str(e))
                        continue  # Skip failed chunks
                    except Exception as e:
                        logging.error("Unexpected error processing chunk %d: %s", idx + 1, str(e))
                        logging.debug(traceback.format_exc())
                        continue  # Skip failed chunks

            logging.info("Number of partial responses: %d", len(partial_responses))

            if not partial_responses:
                return "No matches found for the query"

            # Summarize all partial responses into a final response
            final_response = self.reduce_function(partial_responses, max_tokens=10000)  # Adjust max_tokens as needed
            return final_response

        except LLMPipelineError:
            return "An error occurred while processing your query. Please try again later."
        except Exception as e:
            logging.error("Error in process_query: %s", str(e))
            logging.debug(traceback.format_exc())
            return "An unexpected error occurred. Please contact support."