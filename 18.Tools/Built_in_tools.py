from langchain_community.tools import DuckDuckGoSearchRun, ShellTool

search_tool = DuckDuckGoSearchRun()

search_result = search_tool.invoke("What is the ipl news?")

# print("Search result successfully.", search_result)



shell_tool = ShellTool()

shell_result = shell_tool.invoke("whoami")
print("Shell result successfully.", shell_result)


