from openai import OpenAI
from dotenv import load_dotenv
import os
import research_tools as rt
import json
from datetime import datetime
import re

load_dotenv()

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.getenv("NVIDIA_API_KEY"),
)

def word_count(text: str):
    """
    A function to count the number of words in a given text input.

    Arguments:
        text (str): Input text for which the word count needs to be found

    Returns:
        len(matchobject) (int): Word count found using regular expression checking for whitespace characters.
    """
    matchobject = re.findall("\bw+\b", text)
    return len(matchobject)

def research_agent(prompt:str, model:str = "nvidia/nemotron-3-ultra-550b-a55b", allowed_tools: list = None, max_turns: int = 5):
    """
    A function to carry out a step for the research_agent in the agentic workflow.

    Arguments:
        prompt (str): The text input by the user on which the research has to be carried out.
        model (str): The LLM to be used to carry out the tasks.
        allowed_tools (list): A list of tools that shall be accessed by the LLM at a particular step in the workflow.
        max_turns (int): The number of times the LLM can be called to gather and synthesize information.

    Returns:
        final_answer (str): The final synthesis of information performed by the research agent at this step.
        messages (list[dict]): A list containing detailed logs of each interaction with the LLM and the tools it used.
        sources (list): A list of all the sources gathered by the LLM via tool use in this step.
    """

    print("==============================")
    print("Research Agent")
    print("==============================")

    full_prompt = f"""
	You are an advanced research assistant with expertise in information retrieval and academic research methodology. Your mission is to gather comprehensive, accurate and relevant information on any topic requested by the user.
    
    ## AVAILABLE RESEARCH TOOLS:
	### Mandatory Research Tools:
		1. **`tavily_search_tool`**: General web search engine
	- USE FOR: Recent news, current events, blogs, websites, industry reports, and non-academic sources
	- BEST FOR: Up-to-date information, diverse perspectives, practical applications, and general knowledge

		2. **`arxiv_search_tool`**: Academic publication database
		- USE FOR: Peer-reviewed research papers, technical reports, and scholarly articles
		- LIMITED TO THESE DOMAINS ONLY:
			* Computer Science
			* Mathematics
			* Physics
			* Statistics
			* Quantitative Biology
			* Quantitative Finance
			* Electrical Engineering and Systems Science
			* Economics
		- BEST FOR: Scientific evidence, theoretical frameworks, and technical details in supported fields.

	### Optional Research Tools:
		3. **`wikipedia_search_tool`**: Encyclopedia resource
			- USE FOR: Background information, definitions, overviews, historical context
			- BEST FOR: Establishing foundational knowledge and understanding basic concepts
		
		4. **`semantic_scholar_search_tool`**: Broader academic publication database
		-USE FOR: Peer-reviewed research papers, technical reports and scholarly articles when **the arxiv_search_tool does not yield fruitful results, or when the domain of research is out of what can be searched on arxiv**.
		-Useful domains of research available on semantic scholar but not arxiv:
			* Psychology
			* Literature
			* History
			* Art
			* Architecture
			* Medicine and Life Sciences
			* Sociology
			* Political Science
			* Business and Management
		- BEST FOR: Peer-reviewed coverage across virtually any academic discipline, and using citation counts as a signal of a source's credibility/influence when arXiv's narrower scope doesn't apply.
		
		5. **`scrape_webpage_from_url_list_tool`**: 
			- USE FOR: Getting more content and depth on any links that you find from the tavily_search_tool, arxiv_search_tool, semantic_scholar_search_tool, and wikipedia_search_tool. Use when you deem that enough information has not been captured by you when searching with the other tools, and thus you can use the urls obtained from there to webscrape those sites and extract more information.
			- BEST FOR: Sites that do not block webscrapers and which can easily be scraped.
		
		6. **`run_python_code_tool`**:
			- USE FOR: Running any Python code you generate — not limited to calculations, simulations, or hypothesis testing. Use it whenever executing code would help verify a claim, process or analyze data, or answer a question more reliably than reasoning alone.
				
    ## RESEARCH METHODOLOGY:
    
	1. **ANALYZE REQUEST**: Identify the core research questions and knowledge domains.
    2. **PLAN SEARCH STRATEGY**: Determine which tool is most appropriate for the topic and the instruction given to you
    3. **EXECUTE SEARCHES**: Use the selected tool with effective keywords and queries
    4. **EVALUATE SOURCES**: Prioritize credibility, relevance, recency and diversity
    5. **SYNTHESIZE FINDINGS:**: Organize information logically with clear source attribution
    6. **DOCUMENT SEARCH PROCESS**: Note which tool was used and why
    
    ## TOOL SELECTION GUIDELINES
    
    - For scientific/academic questions in supported domains → Use `arxiv_search_tool`
	- For recent developments, news, or practical information → Use `tavily_search_tool`
    - For non-scientific questions in supported domains → Use `semantic_scholar_search_tool`
	- For fundamental concepts or historical context → Use `wikipedia_search_tool`
    - For obtaining additional information from the sources you have gathered → use `scrape_webpage_from_url_list_tool`
    - For performing calculations/simulations/hypotheses/code execution → Use `run_python_code_tool`
	- NEVER use `arxiv_search_tool` for domains outside its supported list
	- ALWAYS verify information across multiple sources when possible

    For this step YOU NEED NOT CALL EVERY TOOL AVAILABLE TO YOU. ONLY CALL THE TOOLS AS INSTRUCTED.
    For example, if the instruction explicitly tells you to call the `tavily_search_tool`, then YOU CALL ONLY THAT ONE.
    YOU CAN ONLY CALL MULTIPLE TOOLS WHEN THE INSTRUCTION GIVES YOU THE FREEDOM TO DO SO. AND YOU CAN ONLY CALL EACH TOOL AT MOST ONCE.
    When calling a tool, use a single well-chosen query per tool rather than retrying with broader or rephrased queries. Once you have results, synthesize and stop — do not repeat searches to try to get "better" results.
    
    ## OUTPUT FORMAT:

	Present your research findings in a structured format that includes:
	1. **Summary of Research Approach**: Tools used and search strategy
	2. **Key Findings**: Organized by subtopic or source
	3. **Source Details**: Include URLs, titles, authors, and publication dates
	4. **Limitations**: Note any gaps in available information
    
    Today is {datetime.now().strftime("%Y-%m-%d")}.
    """.strip()

    # defining a dictionary of tools to select which ones to provide the LLM with access to
    all_tools = {
    "tavily_search_tool": rt.tavily_search_tool_def,
    "arxiv_search_tool": rt.arxiv_search_tool_def,
    "wikipedia_search_tool": rt.wikipedia_search_tool_def,
    "scrape_webpage_from_url_list_tool": rt.scrape_webpage_from_url_list_tool_def,
    "run_python_code_tool": rt.run_python_code_tool_def,
    "semantic_scholar_search_tool": rt.semantic_scholar_search_tool_def
    }

    # adding access only to the tools allowed by the executor agent
    if allowed_tools:
        tools = [all_tools[name] for name in allowed_tools]
    else:
        tools = list(all_tools.values())
    
    messages = [
        {"role": "system", "content": full_prompt},
        {"role": "user", "content": prompt}
    ]
    

    print(f"max_turns for this call: {max_turns}")

    for i in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0
        )

        # logging the LLM's behavior with each iteration and tool call
        message = response.choices[0].message
        print(f"--- Turn {i} ---")
        print(f"tool_calls: {message.tool_calls}")

        messages.append(message.model_dump())

        if not message.tool_calls:
            print("No tool calls — breaking out of loop.")
            break

        # logging each the information in each tool call performed by the LLM
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            result = rt.tool_mapping[tool_name](**tool_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

    # collecting all the sources used during the research process
    collected_sources = []
    for msg in messages:
        if msg.get("role") == "tool":
            try:
                result = json.loads(msg["content"])
            except json.JSONDecodeError:
                continue

            # checking for a list of dictonaries as a source list
            items = result if isinstance(result, list) else [result]

            for item in items:
                if not isinstance(item, dict):
                    continue

                if "url" in item and "title" in item:
                    collected_sources.append(item)

    print(f"Loop finished.")

    final_answer = message.content

    # handling non-synthesis of information if the LLM kept calling tools for 'max_turns' times
    if final_answer is None:
        print("Loop exhausted mid-tool-call — forcing final synthesis.")
        messages.append({
            "role": "user",
            "content": "Stop calling tools now. Based on everything gathered so far, provide your final research summary as plain text."
        })
        fallback_response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        final_answer = fallback_response.choices[0].message.content or "No research summary could be generated."

    return final_answer, messages, collected_sources

def writer_agent(prompt:str, model="nvidia/nemotron-3-ultra-550b-a55b", min_words_total: int = 1500, retries: int = 1, max_tokens: int = 10000, source_context: str = ""):
    """
        A function to carry out a step for the writer_agent in the agentic workflow.
    
        Arguments:
            prompt (str): The text input by the user on which the research has to be carried out.
            model (str): The LLM to be used to carry out the tasks.
            min_words_total (int): The minimum number of words that the report generated has to exceed in word count.
            retries (int): The amount of times the LLM can 'retry' writing the draft in case it does not match word count
            max_tokens (int): The maximum amount of tokens that can be spent on output by the LLM when writing the draft.
            source_context (str): A formatted string containing all the sources accessed during research to cite them in the right way.
    
        Returns:
            content (str): The draft of the report generated by the writing agent at this step.
            messages (list[dict]): A list containing detailed logs of each interaction with the LLM and the tools it used.
            [] (list): A blank list to match the syntax of the executor_agent function defined in the `planning_agent.py` file.
        """
    

    print("==================================")
    print("Writer Agent")
    print("==================================")

    system_prompt = f"""
    You are an expert academic writer with a PhD-level understanding of scholarly communication. Your task is to synthesize research materials into a comprehensive, well-structured academic report.

    ## REPORT REQUIREMENTS:
    - Produce a COMPLETE, POLISHED and PUBLICATION-READY academic report in markdown format.
    - Create original content that roughly analyzes the provided research materials
    - Do NOT summarize the sources; instead - develop a cohesive narrative with critical analysis
    - Length should be appropriate to thoroughly cover the topic (typically 1500-3000 words)

    ## MANDATORY STRUCTURE:
    1. **Title**: Clear, concise and descriptive of the content
    2. **Abstract**: Brief summary (100-150 words) of the report's purpose, methods and key findings
    3. **Introduction**: Present the topic, research question/problem, significance and outline of the report
    4. **Background/Literature Review**: Contextualize the topic within existing scholarship
    5. **Methodology**: If applicable, describe research methods, data collection and analytical approaches
    6. **Key findings**: Present the primary outcomes and evidence for the same
    7. **Discussion**: Interpret findings, address implications, limitations, and aconnections to the broader field of research
    8. **Conclusion**: Synthesize the main points and suggest directions for future research
    9. **References**: A completed list of all cited works

    ## ACADEMIC WRITING GUIDELINES:
    - Maintain formal, precise, and objective language throughout but a flair for language and communication should be seen
    - Use discipline-appropriate terminology and concepts but do not use jargon that is too nice/deep
    - Support all claims with evidence and reasoning
    - Follow the PEEL structure wherever applicable: point, evidence, explanation, link to RQ
    - Develop a logical flow between ideas, paragraphs and sections. Do not put together too many ideas in one paragraph.
    - Include relevant examples, case studies, data, equations, calculations, simulations, hypotheses or code to strengthen arguments
    - Address potential counterarguments and limitations wherever applicable

    ## CITATION AND REFERENCE RULES:
    - Use numeric inline citations [1], [2], etc. for all borrowed ideas and information
    - Every claim based on external sources MUST have a citation
    - Each inline citation must correspond to a complete entry in the References section
    - Every reference listed must be cited at least once in the text
    - Preserve ALL original URLs, DOIs, and bibliographic information from source materials
    - Format references consistently according to academic standards

    ## FORMATTING GUIDELINES:
    - Use Markdown syntax for all formatting (headings, emphasis, lists, etc.)
    - Include appropriate section headings and subheadings to organize content
    - Format any equations, tables, or figures according to academic conventions
    - Use bullet points or numbered lists when appropriate for clarity
    - Use html syntax to handle all links with target="_blank", so user can always open link in new tab on both html and markdown format

    Output the complete report in Markdown format only. Do not include meta-commentary about the writing process.

    CRITICAL: Only cite sources that appear in the research material provided to you in this conversation. Do NOT cite any book, paper, or source from your own general knowledge, even if you're confident it's real — if it wasn't in the research material you were given, it doesn't belong in this report. If a claim would benefit from a citation you don't have material for, either omit the citation or note it as a claim requiring further verification.

    ## CITATION RULES (STRICT):
    - You will be given a list of available sources, each with a stable tag like [S1], [S2], etc.
    - When citing a source, use ONLY these exact tags, e.g., "...as demonstrated in [S7]."
    - Do NOT write your own References section — one will be generated automatically from the source list. Do NOT list sources, URLs, or bibliographic details anywhere in your output yourself.
    - Do NOT invent, alter, guess at, or reproduce any URL, DOI, author name, or publication detail from memory.
    - CRITICAL: Only cite sources that appear in the provided source list. Do NOT cite any book, paper, author, or source from your own general knowledge, even if you are confident it is real. If a claim would benefit from a citation you were not given a source for, either omit the citation or explicitly note the claim needs further verification — never substitute a source you were not given.
    - If no source list is provided or it's empty, do not fabricate one — write without citations rather than inventing sources.

    ## Available Sources:
    {source_context}

    Cite only using the [S#] tags shown above.

    INTERNAL CHECKLIST (DO NOT INCLUDE IN OUTPUT):
    - [ ] Incorporated all provided research materials
    - [ ] Developed original analysis beyond mere summarization
    - [ ] Included all mandatory sections with appropriate content
    - [ ] Used proper inline citations for all borrowed content
    - [ ] Created complete References section with all cited sources
    - [ ] Maintained academic tone and language throughout
    - [ ] Ensured logical flow and coherent structure
    - [ ] Preserved all source URLs and bibliographic information

    **If the task context includes a section labeled "Feedback" from a prior editor review, treat it as mandatory revision instructions and incorporate every point raised before producing your new draft.**
    """.strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content

    # checking whether the word count is met
    if word_count(content) < min_words_total:
        full_length_content = content
        messages_copy = messages.copy()
        # re-prompting the LLM with feedback to ensure the word count is met if missed in the first try
        for i in range(retries):
            messages_copy += [
                {"role": "assistant", "content": full_length_content},
                {"role": "user", "content": f"Feedback: word count is only {word_count(full_length_content)}, which is less than the minimum required ({min_words_total}). Please expand and rewrite the draft to meet this requirement while maintaining quality and all citation/structure requirements."}
                ]
            response_copy = client.chat.completions.create(
                    model=model,
                    messages=messages_copy,
                    temperature=0,
                    max_tokens=max_tokens,
                )
            full_length_content = response_copy.choices[0].message.content
            if word_count(full_length_content) >= min_words_total:
                break
        return full_length_content, messages, []

    return content, messages, []

def editor_agent(prompt: str, model: str = "nvidia/nemotron-3-ultra-550b-a55b"):
    """
        A function to carry out a step for the editor_agent in the agentic workflow.
    
        Arguments:
            prompt (str): The text input by the user on which the research has to be carried out.
            model (str): The LLM to be used to carry out the tasks.
    
        Returns:
            content (str): The draft of the feedback generated by the editor agent at this step.
            messages (list[dict]): A list containing detailed logs of each interaction with the LLM and the tools it used.
            [] (list): A blank list to match the syntax of the executor_agent function defined in the `planning_agent.py` file.
        """
    
    print("==================================")
    print("Editor Agent")
    print("==================================")

    system_prompt = f"""
    You are a professional academic editor with expertise in improving scholarly articles across disciplines. Your task is to review the academic text provided and either (a) flag it for revision with detailed feedback, or (b) finalize it as a polished, publication-ready document.

    ## PROCESS
    1. Analyze the overall structure, argument flow, coherence, and clarity of the text.
    2. Ensure a logical progression of ideas with clear topic sentences and transitions between paragraphs.
    3. Improve clarity, precision and conciseness of language while maintaining academic tone.
    4. Verify the technical accuracy of the text (to the extent possible based on the context provided to you)
    5. Verify that all in-text citations use the [S#] tag format correctly, and that no claim is supported by a source that doesn't appear in the provided source list.
    6. Look for how the text can enhance readability by cutting jargon that is too deep/niche, and improving the organization and formatting if needed
    7. Incorporate all these elements in a detailed feedback to ensure that the draft can be made ready

    ## Specific Elements to Address:
    - Mention whether thesis statements and main arguments need to be strengthened
    - Mention whether important complex concepts need to be clarified with additional examples or explanations wherever needed
    - Suggest adding relevant equations if needed
    - Check for standardized terminology and eliminate redundancies
    - Focus on improving sentence variety and paragraph structure
    - Focus on preserving citations and maintaining the integrity of the references section
    - Flag any claim that appears to cite a source (book, paper, statistic) not present in the provided source list — this may indicate a fabricated citation and should be treated as a "needs_more_research" issue, not a stylistic one.

    ## DECISION CRITERIA:
    - Set status to "needs_more_research" only if the draft contains claims lacking adequate source support or evidence.
    - Set status to "needs_revision" if there are structural, clarity, or stylistic issues that the writer should address, but the evidence base is sound.
    - Set status to "approved" only if the draft is genuinely publication-ready as-is, or after you personally apply light copy-editing (typo fixes, minor phrasing/flow improvements) that doesn't require the writer to re-draft.

    ## OUTPUT FORMAT (STRICT):
    Respond with ONLY a valid JSON object — no markdown fences, no preamble, no text outside the object. Exactly these keys:
    - "status": "approved", "needs_revision", or "needs_more_research"
    - "feedback": detailed feedback as a single string. Required if status is "needs_revision" or "needs_more_research". Leave as an empty string if status is "approved".
    - "research_gaps": a list of specific, concrete evidence gaps. Only populate if status is "needs_more_research", otherwise an empty list.
    - "final_report": the complete, polished final text, including any light copy-edits you made. Required if and only if status is "approved". Leave as an empty string otherwise.
    """.strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0
    )

    content = response.choices[0].message.content

    return content, messages, []

        


		
    
	

	