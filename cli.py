import requests
import shlex
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# --- Configuration ---
BASE_URL = os.getenv("CHATTY_API_URL", "http://localhost:8000")

# --- Application State ---
APP_STATE = {
    "token": None,
    "user_email": None,
    "docs_context": [],
}

# --- Rich Console ---
console = Console()

# --- Command Handler Functions ---

def handle_help(*args):
    """Displays the help message."""
    table = Table(title="Chatty CLI Commands", show_header=False, box=None)
    table.add_row("[bold cyan]login[/bold cyan]", "Log in to the service to start a session.")
    table.add_row("[bold cyan]register[/bold cyan]", "Register a new user account.")
    table.add_row("[bold cyan]logout[/bold cyan]", "End the current session.")
    table.add_row("[bold cyan]list_docs[/bold cyan]", "List available documents in the knowledge base.")
    table.add_row("[bold cyan]use_docs [file1.txt] ...[/bold cyan]", "Set the context to specific documents for subsequent chat messages.")
    table.add_row("[bold cyan]use_docs *[/bold cyan]", "Shortcut to clear the document context (same as 'clear_docs').")
    table.add_row("[bold cyan]clear_docs[/bold cyan]", "Clear the document context to query all documents again.")
    table.add_row("[bold cyan]help[/bold cyan]", "Show this help message.")
    table.add_row("[bold cyan]exit[/bold cyan]", "Exit the CLI application.")
    table.add_row("[bold cyan]<any other text>[/bold cyan]", "Will be sent as a chat message.")
    console.print(table)

def handle_register(args_str):
    """Registers a new user."""
    userid = console.input("[bold]Enter new User ID: [/bold]")
    email = console.input("[bold]Enter your Email: [/bold]")
    password = console.input("[bold]Enter Password: [/bold]", password=True)
    
    if not all([userid, email, password]):
        console.print("[bold red]All fields are required.[/bold red]")
        return
        
    response = requests.post(f"{BASE_URL}/register", json={"userid": userid, "emailid": email, "password": password})
    if response.status_code == 200:
        console.print("[bold green]✔ Registration successful! Please use 'login' to continue.[/bold green]")
    else:
        console.print(f"[bold red]❌ Registration failed:[/bold red] {response.json().get('detail', 'Unknown error')}")

def handle_login(args_str):
    """Logs in and saves token to app state."""
    email = console.input("[bold]Email: [/bold]")
    password = console.input("[bold]Password: [/bold]", password=True)
    if not all([email, password]):
        console.print("[bold red]Email and password are required.[/bold red]")
        return
        
    try:
        response = requests.post(f"{BASE_URL}/login", data={"username": email, "password": password})
        if response.status_code == 200:
            APP_STATE["token"] = response.json()["access_token"]
            APP_STATE["user_email"] = email
            console.print("[bold green]✔ Login successful.[/bold green]")
        else:
            console.print(f"[bold red]❌ Login failed:[/bold red] {response.json().get('detail', 'Invalid credentials')}")
    except requests.exceptions.RequestException:
        console.print(f"[bold red]Connection Error:[/bold red] Could not connect to the API at {BASE_URL}.")

def handle_logout(*args):
    """Logs out by clearing the session state."""
    APP_STATE["token"] = None
    APP_STATE["user_email"] = None
    APP_STATE["docs_context"] = []
    console.print("[bold yellow]You have been logged out.[/bold yellow]")

def handle_list_docs(*args):
    """Lists available documents."""
    if not APP_STATE["token"]:
        console.print("[bold red]You must be logged in first.[/bold red]")
        return
    headers = {"Authorization": f"Bearer {APP_STATE['token']}"}
    response = requests.get(f"{BASE_URL}/documents", headers=headers)
    if response.status_code == 200:
        docs = response.json().get("documents", [])
        if not docs:
            console.print("[yellow]No documents found in the knowledge base.[/yellow]")
            return
        table = Table("Available Knowledge Base Documents")
        for doc in sorted(docs):
            table.add_row(doc)
        console.print(table)
    else:
        console.print(f"[bold red]Error fetching documents: {response.json().get('detail', 'Unknown error')}[/bold red]")
        
def handle_clear_docs(*args):
    """Clears the document context."""
    APP_STATE["docs_context"] = []
    console.print("[bold yellow]Document context cleared. Chat will now query all documents.[/bold yellow]")

# --- UPDATED FUNCTION ---
def handle_use_docs(args_str):
    """
    Sets the document context for the next chat.
    Using '*' as the argument will clear the context.
    """
    if not args_str:
        console.print("[bold red]Usage: use_docs [file1.pdf] [file2.txt] ... or use_docs * to clear.[/bold red]")
        return

    # Check for the special '*' wildcard to clear the context
    if args_str.strip() == '*':
        handle_clear_docs() # Re-use the existing clear function
        return # Exit the function

    # If not '*', proceed with the normal file processing
    try:
        # Use shlex to handle filenames with spaces if they are quoted
        APP_STATE["docs_context"] = shlex.split(args_str)
        console.print(f"[bold yellow]Document context set to: {APP_STATE['docs_context']}. The next chat will only query these files.[/bold yellow]")
    except ValueError as e:
        console.print(f"[bold red]Error parsing filenames (check for unmatched quotes): {e}[/bold red]")


def handle_chat(message):
    """Sends a message to the chatbot."""
    if not APP_STATE["token"]:
        console.print("[bold red]You must be logged in to chat.[/bold red]")
        return
        
    headers = {"Authorization": f"Bearer {APP_STATE['token']}"}
    payload = {"message": message}
    if APP_STATE["docs_context"]:
        payload["source_files"] = APP_STATE["docs_context"]

    try:
        with console.status("[bold green]Thinking...[/bold green]"):
            response = requests.post(f"{BASE_URL}/chat", headers=headers, json=payload)
        
        if response.status_code == 200:
            ai_response = response.json().get("response", "No response content.")
            console.print(Panel(ai_response, title="Chatbot", border_style="magenta", title_align="left"))
        else:
            console.print(f"[bold red]Error: {response.json().get('detail', 'Unknown error')}[/bold red]")
    except requests.exceptions.RequestException:
        console.print(f"[bold red]Connection Error:[/bold red] Could not connect to the API at {BASE_URL}.")

# --- Command Dispatcher ---
COMMANDS = {
    "help": handle_help,
    "register": handle_register,
    "login": handle_login,
    "logout": handle_logout,
    "list_docs": handle_list_docs,
    "use_docs": handle_use_docs,
    "clear_docs": handle_clear_docs,
    "exit": lambda *args: exit(),
    "quit": lambda *args: exit(),
}

def get_current_prompt():
    """Generates the dynamic prompt string."""
    user_part = APP_STATE.get("user_email", "logged out")
    docs_part = ""
    if APP_STATE.get("docs_context"):
        num_docs = len(APP_STATE['docs_context'])
        docs_part = f" [{num_docs} doc{'s' if num_docs > 1 else ''}]"
    
    return f"chatty ({user_part}){docs_part} > "

def main():
    """The main interactive loop."""
    console.print("[bold]Welcome to the Chatty CLI![/bold] Type 'help' for a list of commands.")
    while True:
        try:
            prompt_text = get_current_prompt()
            user_input = console.input(prompt_text).strip()

            if not user_input:
                continue

            parts = user_input.split(' ', 1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if command in COMMANDS:
                COMMANDS[command](args)
            else:
                handle_chat(user_input)
        
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold]Exiting...[/bold]")
            break
        except SystemExit:
            console.print("\n[bold]Exiting...[/bold]")
            break

if __name__ == "__main__":
    main()