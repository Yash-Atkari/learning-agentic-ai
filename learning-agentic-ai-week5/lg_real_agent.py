import os
from dotenv import load_dotenv
import smtplib
import imaplib
import email
from email.header import decode_header
from email.mime.text import MIMEText

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

# Langgraph imports
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

GMAIL_USER = "yashatkari7@gmail.com"
GMAIL_PASS = "fnaitoboxfdwhcyy"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY")
)

# Define the real tools
@tool 
def check_real_gmail(limit: int = 3) -> str:
    """Connects to real Gmail and reads the latest N email subjects."""
    print(f"(Real) Connecting to Gmail as {GMAIL_USER}...")
    try:
        # Connect to Gmail IMAP
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(GMAIL_USER, GMAIL_PASS)
        mail.select("inbox")

        # Search for all emails
        status, messages = mail.search(None, "ALL")
        email_ids = messages[0].split()

        # Get the latest N emails
        latest_emails = email_ids[-limit:]

        results = []
        for e_id in latest_emails:
            # Fetch the email
            _, msg_data = mail.fetch(e_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject, encoding  = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding if encoding else "utf-8")
                    from_ = msg.get("From")
                    results.append(f"- From: {from_} | Subject: {subject}")
        
        mail.logout()
        return "\n".join(results)
    
    except Exception as e:
        return f"Error reading email: {str(e)}"
    
@tool
def send_real_email(to_email: str, subject: str, body: str) -> str:
    """Sends the real email to the specified address."""
    print(f"(REAL) Sending email to {to_email}...")
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = GMAIL_USER
        msg['To'] = to_email

        # Connect to Gmail SMTP
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(GMAIL_USER, GMAIL_PASS)
            smtp_server.sendmail(GMAIL_USER, to_email, msg.as_string())
        
        return "Success: Email sent successfully."
    except Exception as e:
        return f"Error sending email: {str(e)}"

# Bind tool
tools = [check_real_gmail, send_real_email]
llm_with_tools = llm.bind_tools(tools)

# Define the nodes
def agent_node(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# Build the graph
workflow = StateGraph(MessagesState)

# Add node & edges
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile()

print("Real Gmail Agent is online...")

user_task = "Check my last 3 emails and summarize who they are from. Send an email to wonkargaykwad@gmail.com asking about he will receive the acknowdgement letter of the NGO from college or not."

intial_state = {
    "messages": [
        SystemMessage(content="You are a helpful assistant capable of reading and sending real emails."),
        HumanMessage(content=user_task)
    ]
}

for event in app.stream(intial_state):
    for key, value in event.items():
        print(f"Current node: {key}")
        last_msg = value["messages"][-1]
        if last_msg.content:
            print(f"Message: {last_msg.content}\n")
        else:
            print(f"Message: Agent want to call {len(last_msg.tool_calls)} tools\n")

