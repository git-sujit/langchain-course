#import os
#from dotenv import load_dotenv
#load_dotenv(override=True)

from langchain_core.prompts import ChatPromptTemplate
#from langchain_openai import ChatOpenAI

from langchain_ollama import ChatOllama

def main():
    print("Hello from langchain-course, Local Models!")
    # print(os.getenv("OPENAI_API_KEY"))
    information = """
    Andrew R. Jassy (born January 13, 1968)[5] is an American business executive who is the president and chief executive officer of Amazon since July 2021, succeeding founder Jeff Bezos, \
        who remains executive chairman.[6][7] Jassy founded and led Amazon Web Services (AWS) from its inception and served as its CEO from April 2016 until July 2021.[8] Jassy is the son of Margery and Everett L. Jassy of Scarsdale, New York.[3] Of Jewish[9][10] Hungarian ancestry,[9] his father was a senior partner in the corporate law firm Dewey Ballantine in New York City, and chairman of the firm's management committee.[3] Jassy grew up in Scarsdale, and attended Scarsdale High School,[3][11] where he played varsity soccer and tennis.[12] Jassy graduated cum laude from Harvard College in government, where he was advertising manager of The Harvard Crimson. He later earned an MBA from Harvard Business School. In 1989, he wrote in The Crimson that the newspaper should continue to publish advertisements from Eastern Air Lines, despite an ongoing labor dispute there.[13][14][15] \
        Jassy joined Amazon as a marketing manager in 1997.[12] Early in his Amazon career, he helped run the company's first marketing team and later its compact disc business.[17] He subsequently served as Jeff Bezos's first technical adviser, or shadow, a role in which he accompanied Bezos to meetings and discussed potential business opportunities with him.[17] In 2003, he and Bezos came up with the idea to create the cloud computing platform that became known as Amazon Web Services (AWS), which launched in 2006.[18] Jassy headed AWS and its original team of 57 people.[1]
    """
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant that can answer questions and help with tasks. Given the {information}:"
                1. A short Summary of the information
                2. Two interesting facts about the person mentioned in the information"""
    )
    print("--------------------------------")
    print("***** Google Gemma3:270m *****")
    model_name = "gemma3:270m"
    model = ChatOllama(model=model_name, temperature=0)
    chain = prompt | model
    response = chain.invoke({"information": information})
    print(response.content)
    print("--------------------------------")
    print("***** Qwen3.5:9b *****")
    model_name = "qwen3.5:9b"
    model = ChatOllama(model=model_name, temperature=0)
    chain = prompt | model
    response = chain.invoke({"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
