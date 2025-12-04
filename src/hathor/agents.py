import autogen
import arxiv
# import requests
# from io import BytesIO
# import fitz
import numpy as np

class ResearcherAgent(autogen.AssistantAgent):
    """Custom agent with private literature knowledge."""
    
    def __init__(self, name, system_message, llm_config, papers_per_query=10):
        if "tools" not in llm_config:
                llm_config["tools"] = []
            
        llm_config["tools"].append({
            "type": "function",
            "function": {
                "name": "search_arxiv",
                "description": "Search arXiv for academic papers on a given topic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for arXiv papers"
                        }
                    },
                    "required": ["query"]
                }
            }
        })
        
        super().__init__(name, system_message, llm_config)
        self.papers = []
        self._base_system_message = system_message
        self.papers_per_query = papers_per_query

        def search_arxiv_tool(query:str = None) -> str:
            """Search arXiv for academic papers."""
            if query is None:
                query = "Galxy formation"
            return self._search_arxiv(query)
        
        autogen.register_function(
            search_arxiv_tool,
            caller=self,
            executor=self,
            name="search_arxiv",
            description="Search arXiv for papers and add them to the agent's knowledge base."
        )
        

    def _search_arxiv(self, query:str):
        """
            Search arxiv using the given query, and add random subset of papers to self.papers. 
            The query must be simple, as arxiv cannot accommodate complex inputs. 
            
            Example queries:
            - "galaxy formation"
            - "circumgalactic medium"
            - "AGN feedback"
            
            Args:
                query: Simple search query for arxiv
                total: Number of papers to retrieve (default 10)
                
            Returns:
                String describing papers added`
        """
        half = self.papers_per_query//2
        client = arxiv.Client()
        search = arxiv.Search(query=query,  
                                max_results=20,
                                sort_by=arxiv.SortCriterion.SubmittedDate,
                                sort_order=arxiv.SortOrder.Descending 
                            )
        results = list(client.results(search))
        recent_papers = np.random.choice(results, size=5, replace=False)

        search = arxiv.Search(query=query,  
                                max_results=20,
                                sort_by=arxiv.SortCriterion.Relevance, 
                                sort_order=arxiv.SortOrder.Descending
                            )
        results = list(client.results(search))
        relevant_papers = np.random.choice(results, size=5, replace=False)
        chosen_papers = np.concatenate([recent_papers,relevant_papers])

        paper_strings = self._get_paper_strings(chosen_papers)

        self._add_papers(paper_strings)

        return "/n/n/n".join(paper_strings)
        
    def _get_paper_strings(self, papers):
        paper_strings = []
        for paper in papers:
            # Download and extract full text
            # try:
            #     response = requests.get(paper.pdf_url)
            #     pdf_file = BytesIO(response.content)
                
            #     doc = fitz.open(stream=pdf_file, filetype="pdf")
            #     full_text = ""
            #     for page in doc:
            #         full_text += page.get_text()
            #     doc.close()
            # except Exception as e:
            #     full_text = f"[Error extracting text: {e}]"
            full_text = ""
            
            # Create paper string with correct attributes
            paper_string = f"""
                            Title: {paper.title}
                            Abstract: {paper.summary}
                            Full Text: {full_text}
                    """
            
            paper_strings.append(paper_string)

        return paper_strings

    def _add_papers(self, paper_strings):
        self.papers.extend(paper_strings)

        new_system_message = self._base_system_message
        new_system_message += "\n\n" + "="*80 + "\n"
        new_system_message += "SAVED PAPERS:\n"
        new_system_message += "="*80 + "\n\n"
        
        for paper in paper_strings:
            new_system_message += paper
            new_system_message += "="*80 + "\n\n"

        self.update_system_message(new_system_message)