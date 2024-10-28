import json
import requests
import schedule
import time
import threading


#########################
# Variables obligatoires
#########################
OUTPUT_FILE = "commits.json"  # Extraire les commits dans ce fichier
GITHUB_PAT = "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"  # Genérez un token dans github et collez le ici


GITHUB_URL = "https://api.github.com/repos"
REPO = "hive"
OWNER = "apache"

HEADERS = {
	"Accept": "application/vnd.github+json",
	"Authorization": "token " + GITHUB_PAT
}


def extract_commits(page: int = 1, per_page: int = 100) -> list[dict]:
    """Extract a list of commits.

    Args:
        page (int, optional): The page number of the results to fetch. Defaults to 1.
        per_page (int, optional): The number of results per page (max 100). Defaults to 100.

    Returns:
        list[dict]: List of dictionnaries containning:
            - "message": The message of the commit.
            - "sha": Commit hash. 
    """
    try:
        response = requests.get(f"{GITHUB_URL}/{OWNER}/{REPO}/commits?per_page={per_page}&page={page}", headers=HEADERS)
        response.raise_for_status()
        return [{"sha": e["sha"], "message": e["commit"]["message"]} for e in json.loads(response.text)]
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"An error occurred while fetching commits:\n\n\t\t{e}\n")
        return {}


def extract_commit_content(hash: str) -> dict:
    """Extract the content of a single commit.

    Args:
        hash (str): Commit reference.

    Returns:
        dict: Dictionnary containning:
            - "message": The message of the commit.
            - "files": List of modified Java filenames. 
    """
    try:
        response = requests.get(f"{GITHUB_URL}/{OWNER}/{REPO}/commits/" + hash, headers=HEADERS)
        commit = json.loads(response.text)
    except (requests.exceptions.RequestException, json.JSONDecodeError):
        return {}
    return {
        "message": commit["commit"]["message"],
        "files": [e["filename"] for e in commit["files"] if e["filename"].endswith(".java") and e["status"] == "modified"]
    }


def _print_commit_count():
    print(f"Found {round(len(found)*100/len(tickets), 2)}% commits.")


def _start_scheduler():
    schedule.every(10).seconds.do(_print_commit_count)
    while not stop_event.is_set():
        schedule.run_pending()
        time.sleep(1)


stop_event = threading.Event()
scheduler_thread = threading.Thread(target=_start_scheduler)
scheduler_thread.start()

tickets = [  # Tickets de la version 4.0.0 de jira
    "HIVE-18284", "HIVE-24849", "HIVE-21778", "HIVE-26018", "HIVE-22170", "HIVE-25268", "HIVE-20607", "HIVE-20209", "HIVE-20192", "HIVE-19886", "HIVE-23339",
    "HIVE-24570", "HIVE-25659", "HIVE-24944", "HIVE-24606", "HIVE-24579", "HIVE-28347", "HIVE-27649", "HIVE-25299", "HIVE-25646", "HIVE-26036", "HIVE-25912",
    "HIVE-24020", "HIVE-24902", "HIVE-25626", "HIVE-25795", "HIVE-25577", "HIVE-24851", "HIVE-25777", "HIVE-28262", "HIVE-21301", "HIVE-20542", "HIVE-20541",
    "HIVE-20583", "HIVE-21624", "HIVE-21992", "HIVE-21825", "HIVE-24523", "HIVE-21391", "HIVE-28239", "HIVE-28121", "HIVE-26339", "HIVE-28166", "HIVE-28439",
    "HIVE-28173", "HIVE-28487", "HIVE-28515", "HIVE-28266", "HIVE-28207", "HIVE-28451", "HIVE-28264", "HIVE-28202", "HIVE-28278", "HIVE-28143", "HIVE-28285",
    "HIVE-28006", "HIVE-28366", "HIVE-28330", "HIVE-25758", "HIVE-20016", "HIVE-26778", "HIVE-28009", "HIVE-27801", "HIVE-25803", "HIVE-28123", "HIVE-22961",
    "HIVE-24147", "HIVE-21976", "HIVE-20632", "HIVE-21344", "HIVE-22336", "HIVE-22824", "HIVE-21564", "HIVE-26017", "HIVE-26767", "HIVE-27063", "HIVE-26911",
    "HIVE-27071", "HIVE-24928", "HIVE-25547", "HIVE-24809", "HIVE-20680", "HIVE-22826", "HIVE-26158", "HIVE-22360", "HIVE-24786", "HIVE-21729", "HIVE-25384",
    "HIVE-26415", "HIVE-24322", "HIVE-20627", "HIVE-21261", "HIVE-21286", "HIVE-20682", "HIVE-21186", "HIVE-20805", "HIVE-20817", "HIVE-21281", "HIVE-20511",
    "HIVE-21260", "HIVE-20911", "HIVE-20631", "HIVE-21269", "HIVE-20629", "HIVE-20953", "HIVE-21206", "HIVE-21722", "HIVE-22107", "HIVE-21811", "HIVE-21446",
    "HIVE-21325", "HIVE-21471", "HIVE-21892", "HIVE-22537", "HIVE-23358", "HIVE-22080", "HIVE-21403", "HIVE-22708", "HIVE-22555", "HIVE-22988", "HIVE-21706",
    "HIVE-21307", "HIVE-23408", "HIVE-21306", "HIVE-21654", "HIVE-23345", "HIVE-22110", "HIVE-21694", "HIVE-21717", "HIVE-22903", "HIVE-22663", "HIVE-21776",
    "HIVE-21700", "HIVE-21730", "HIVE-22877", "HIVE-21421", "HIVE-23925", "HIVE-23962", "HIVE-25242", "HIVE-23851", "HIVE-24293", "HIVE-25957", "HIVE-25749",
    "HIVE-25825", "HIVE-24876", "HIVE-24030", "HIVE-24345", "HIVE-25582", "HIVE-24882", "HIVE-24951", "HIVE-24803", "HIVE-23582", "HIVE-24097", "HIVE-25746",
    "HIVE-25917", "HIVE-25914", "HIVE-24213", "HIVE-25964", "HIVE-25986", "HIVE-25716", "HIVE-24501", "HIVE-25117", "HIVE-25926", "HIVE-23887", "HIVE-23763",
    "HIVE-25150", "HIVE-25734", "HIVE-24068", "HIVE-25570", "HIVE-25973", "HIVE-25757", "HIVE-25009", "HIVE-24256", "HIVE-24530", "HIVE-24550", "HIVE-25766",
    "HIVE-24957", "HIVE-25675", "HIVE-24751", "HIVE-25774", "HIVE-25163", "HIVE-25530", "HIVE-24446", "HIVE-25621", "HIVE-26268", "HIVE-25907", "HIVE-25408",
    "HIVE-25538", "HIVE-25386", "HIVE-25316", "HIVE-24514", "HIVE-24691", "HIVE-24198", "HIVE-22290", "HIVE-21437", "HIVE-21880", "HIVE-22313", "HIVE-22955"
]

found = {}  # Tickets trouvés dans github

max_tries = 20
min_found = int(0.9 * len(tickets))
try_count = 0
previous_found_count = 0

page = 1
per_page = 100
commits = extract_commits(page=page, per_page=per_page)

while commits or len(found) != len(tickets):

    # Pour chaque ticket chercher le numéro de commit associé
    for commit in commits:
        message = commit["message"]
        for ticket in tickets:
            if ticket in message:
                found[ticket] = commit
                break

    if len(found) == previous_found_count and len(found) > min_found:
        try_count += 1
        if try_count >= max_tries:
            print(f"Max tries reached... {len(tickets) - len(found)} ticket(s) not found.")
            break

    else:
        try_count = 0
        previous_found_count = len(found)

    page += 1
    commits = extract_commits(page=page, per_page=100)

if found:
    with open(OUTPUT_FILE, "w") as file:
        json.dump(found, file, indent=4)

stop_event.set()
scheduler_thread.join()
