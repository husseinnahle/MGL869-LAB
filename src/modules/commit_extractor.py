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
            - "files": List of modified Java and C++ filenames. 
    """
    try:
        response = requests.get(f"{GITHUB_URL}/{OWNER}/{REPO}/commits/" + hash, headers=HEADERS)
        commit = json.loads(response.text)
    except (requests.exceptions.RequestException, json.JSONDecodeError):
        return {}
    return [e["filename"] for e in commit["files"] \
            if (e["filename"].endswith(".java") or e["filename"].endswith(".cpp")) and e["status"] == "modified"]


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

# tickets = [  # Tickets de la version 4.0.0 de jira
#     "HIVE-18284", "HIVE-24849", "HIVE-21778", "HIVE-26018", "HIVE-22170", "HIVE-25268", "HIVE-20607", "HIVE-20209", "HIVE-20192", "HIVE-19886", "HIVE-23339",
#     "HIVE-24570", "HIVE-25659", "HIVE-24944", "HIVE-24606", "HIVE-24579", "HIVE-28347", "HIVE-27649", "HIVE-25299", "HIVE-25646", "HIVE-26036", "HIVE-25912",
#     "HIVE-24020", "HIVE-24902", "HIVE-25626", "HIVE-25795", "HIVE-25577", "HIVE-24851", "HIVE-25777", "HIVE-28262", "HIVE-21301", "HIVE-20542", "HIVE-20541",
#     "HIVE-20583", "HIVE-21624", "HIVE-21992", "HIVE-21825", "HIVE-24523", "HIVE-21391", "HIVE-28239", "HIVE-28121", "HIVE-26339", "HIVE-28166", "HIVE-28439",
#     "HIVE-28173", "HIVE-28487", "HIVE-28515", "HIVE-28266", "HIVE-28207", "HIVE-28451", "HIVE-28264", "HIVE-28202", "HIVE-28278", "HIVE-28143", "HIVE-28285",
#     "HIVE-28006", "HIVE-28366", "HIVE-28330", "HIVE-25758", "HIVE-20016", "HIVE-26778", "HIVE-28009", "HIVE-27801", "HIVE-25803", "HIVE-28123", "HIVE-22961",
#     "HIVE-24147", "HIVE-21976", "HIVE-20632", "HIVE-21344", "HIVE-22336", "HIVE-22824", "HIVE-21564", "HIVE-26017", "HIVE-26767", "HIVE-27063", "HIVE-26911",
#     "HIVE-27071", "HIVE-24928", "HIVE-25547", "HIVE-24809", "HIVE-20680", "HIVE-22826", "HIVE-26158", "HIVE-22360", "HIVE-24786", "HIVE-21729", "HIVE-25384",
#     "HIVE-26415", "HIVE-24322", "HIVE-20627", "HIVE-21261", "HIVE-21286", "HIVE-20682", "HIVE-21186", "HIVE-20805", "HIVE-20817", "HIVE-21281", "HIVE-20511",
#     "HIVE-21260", "HIVE-20911", "HIVE-20631", "HIVE-21269", "HIVE-20629", "HIVE-20953", "HIVE-21206", "HIVE-21722", "HIVE-22107", "HIVE-21811", "HIVE-21446",
#     "HIVE-21325", "HIVE-21471", "HIVE-21892", "HIVE-22537", "HIVE-23358", "HIVE-22080", "HIVE-21403", "HIVE-22708", "HIVE-22555", "HIVE-22988", "HIVE-21706",
#     "HIVE-21307", "HIVE-23408", "HIVE-21306", "HIVE-21654", "HIVE-23345", "HIVE-22110", "HIVE-21694", "HIVE-21717", "HIVE-22903", "HIVE-22663", "HIVE-21776",
#     "HIVE-21700", "HIVE-21730", "HIVE-22877", "HIVE-21421", "HIVE-23925", "HIVE-23962", "HIVE-25242", "HIVE-23851", "HIVE-24293", "HIVE-25957", "HIVE-25749",
#     "HIVE-25825", "HIVE-24876", "HIVE-24030", "HIVE-24345", "HIVE-25582", "HIVE-24882", "HIVE-24951", "HIVE-24803", "HIVE-23582", "HIVE-24097", "HIVE-25746",
#     "HIVE-25917", "HIVE-25914", "HIVE-24213", "HIVE-25964", "HIVE-25986", "HIVE-25716", "HIVE-24501", "HIVE-25117", "HIVE-25926", "HIVE-23887", "HIVE-23763",
#     "HIVE-25150", "HIVE-25734", "HIVE-24068", "HIVE-25570", "HIVE-25973", "HIVE-25757", "HIVE-25009", "HIVE-24256", "HIVE-24530", "HIVE-24550", "HIVE-25766",
#     "HIVE-24957", "HIVE-25675", "HIVE-24751", "HIVE-25774", "HIVE-25163", "HIVE-25530", "HIVE-24446", "HIVE-25621", "HIVE-26268", "HIVE-25907", "HIVE-25408",
#     "HIVE-25538", "HIVE-25386", "HIVE-25316", "HIVE-24514", "HIVE-24691", "HIVE-24198", "HIVE-22290", "HIVE-21437", "HIVE-21880", "HIVE-22313", "HIVE-22955"
# ]

tickets = [  # Tickets de la version 2.0.0 de jira
    "HIVE-22412", "HIVE-22196", "HIVE-21033", "HIVE-20345", "HIVE-18742", "HIVE-18735", "HIVE-18624", "HIVE-18250", "HIVE-18090", "HIVE-17368", "HIVE-16769", "HIVE-16654",
    "HIVE-16507", "HIVE-16450", "HIVE-16385", "HIVE-15769", "HIVE-15766", "HIVE-15752", "HIVE-15517", "HIVE-15329", "HIVE-15311", "HIVE-15297", "HIVE-15247", "HIVE-15234",
    "HIVE-15160", "HIVE-15096", "HIVE-15054", "HIVE-14393", "HIVE-14367", "HIVE-14342", "HIVE-14324", "HIVE-14322", "HIVE-14296", "HIVE-14251", "HIVE-14241", "HIVE-14236",
    "HIVE-14229", "HIVE-14210", "HIVE-14153", "HIVE-14132", "HIVE-14038", "HIVE-14015", "HIVE-14006", "HIVE-13989", "HIVE-13932", "HIVE-13858", "HIVE-13844", "HIVE-13837",
    "HIVE-13833", "HIVE-13813", "HIVE-13809", "HIVE-13756", "HIVE-13754", "HIVE-13725", "HIVE-13704", "HIVE-13699", "HIVE-13693", "HIVE-13676", "HIVE-13619", "HIVE-13618",
    "HIVE-13608", "HIVE-13602", "HIVE-13570", "HIVE-13561", "HIVE-13553", "HIVE-13542", "HIVE-13523", "HIVE-13512", "HIVE-13463", "HIVE-13456", "HIVE-13452", "HIVE-13440",
    "HIVE-13439", "HIVE-13428", "HIVE-13423", "HIVE-13410", "HIVE-13405", "HIVE-13395", "HIVE-13394", "HIVE-13390", "HIVE-13383", "HIVE-13381", "HIVE-13373", "HIVE-13361",
    "HIVE-13335", "HIVE-13333", "HIVE-13330", "HIVE-13326", "HIVE-13320", "HIVE-13300", "HIVE-13299", "HIVE-13293", "HIVE-13286", "HIVE-13285", "HIVE-13261", "HIVE-13260", 
    "HIVE-13240", "HIVE-13236", "HIVE-13217", "HIVE-13216", "HIVE-13201", "HIVE-13200", "HIVE-13191", "HIVE-13186", "HIVE-13175", "HIVE-13174", "HIVE-13169", "HIVE-13163",
    "HIVE-13159", "HIVE-13151", "HIVE-13144", "HIVE-13141", "HIVE-13134", "HIVE-13126", "HIVE-13108", "HIVE-13105", "HIVE-13101", "HIVE-13096", "HIVE-13094", "HIVE-13083",
    "HIVE-13082", "HIVE-13071", "HIVE-13065", "HIVE-13062", "HIVE-13042", "HIVE-13039", "HIVE-13036", "HIVE-13032", "HIVE-13024", "HIVE-12999", "HIVE-12996", "HIVE-12993",
    "HIVE-12992", "HIVE-12981", "HIVE-12964", "HIVE-12947", "HIVE-12945", "HIVE-12937", "HIVE-12933", "HIVE-12931", "HIVE-12927", "HIVE-12926", "HIVE-12911", "HIVE-12909",
    "HIVE-12905", "HIVE-12904", "HIVE-12893", "HIVE-12879", "HIVE-12864", "HIVE-12826", "HIVE-12824", "HIVE-12820", "HIVE-12815", "HIVE-12809", "HIVE-12808", "HIVE-12800",
    "HIVE-12797", "HIVE-12786", "HIVE-12784", "HIVE-12768", "HIVE-12766", "HIVE-12762", "HIVE-12758", "HIVE-12743", "HIVE-12741", "HIVE-12740", "HIVE-12735", "HIVE-12734",
    "HIVE-12728", "HIVE-12726", "HIVE-12725", "HIVE-12724", "HIVE-12712", "HIVE-12708", "HIVE-12698", "HIVE-12694", "HIVE-12688", "HIVE-12687", "HIVE-12685", "HIVE-12684",
    "HIVE-12662", "HIVE-12657", "HIVE-12643", "HIVE-12633", "HIVE-12609", "HIVE-12601", "HIVE-12599", "HIVE-12596", "HIVE-12590", "HIVE-12584", "HIVE-12578", "HIVE-12577",
    "HIVE-12542", "HIVE-12537", "HIVE-12532", "HIVE-12509", "HIVE-12503", "HIVE-12500", "HIVE-12498", "HIVE-12491", "HIVE-12490", "HIVE-12487", "HIVE-12479", "HIVE-12478",
    "HIVE-12477", "HIVE-12473", "HIVE-12472", "HIVE-12465", "HIVE-12463", "HIVE-12456", "HIVE-12451", "HIVE-12450", "HIVE-12445", "HIVE-12435", "HIVE-12407", "HIVE-12405",
    "HIVE-12399", "HIVE-12396", "HIVE-12391", "HIVE-12385", "HIVE-12378", "HIVE-12364", "HIVE-12363", "HIVE-12349", "HIVE-12344", "HIVE-12332", "HIVE-12330", "HIVE-12315",
    "HIVE-12304", "HIVE-12302", "HIVE-12281", "HIVE-12261", "HIVE-12260", "HIVE-12257", "HIVE-12246", "HIVE-12238", "HIVE-12234", "HIVE-12223", "HIVE-12215", "HIVE-12210",
    "HIVE-12208", "HIVE-12207", "HIVE-12204", "HIVE-12198", "HIVE-12189", "HIVE-12171", "HIVE-12166", "HIVE-12090", "HIVE-12058", "HIVE-12039", "HIVE-12021", "HIVE-12008",
    "HIVE-11980", "HIVE-11975", "HIVE-11966", "HIVE-11954", "HIVE-11945", "HIVE-11932", "HIVE-11928", "HIVE-11922", "HIVE-11889", "HIVE-11875", "HIVE-11863", "HIVE-11860",
    "HIVE-11843", "HIVE-11842", "HIVE-11841", "HIVE-11838", "HIVE-11835", "HIVE-11832", "HIVE-11826", "HIVE-11824", "HIVE-11821", "HIVE-11820", "HIVE-11813", "HIVE-11792",
    "HIVE-11781", "HIVE-11761", "HIVE-11751", "HIVE-11745", "HIVE-11726", "HIVE-11723", "HIVE-11721", "HIVE-11710", "HIVE-11705", "HIVE-11698", "HIVE-11696", "HIVE-11669",
    "HIVE-11658", "HIVE-11652", "HIVE-11604", "HIVE-11602", "HIVE-11596", "HIVE-11594", "HIVE-11592", "HIVE-11590", "HIVE-11581", "HIVE-11578", "HIVE-11573", "HIVE-11556",
    "HIVE-11546", "HIVE-11541", "HIVE-11511", "HIVE-11498", "HIVE-11497", "HIVE-11493", "HIVE-11476", "HIVE-11472", "HIVE-11468", "HIVE-11462", "HIVE-11452", "HIVE-11451",
    "HIVE-11434", "HIVE-11433", "HIVE-11430", "HIVE-11428", "HIVE-11425", "HIVE-11413", "HIVE-11406", "HIVE-11397", "HIVE-11375", "HIVE-11312", "HIVE-11303", "HIVE-11293",
    "HIVE-11284", "HIVE-11258", "HIVE-11255", "HIVE-11230", "HIVE-11229", "HIVE-11228", "HIVE-11221", "HIVE-11215", "HIVE-11202", "HIVE-11198", "HIVE-11194", "HIVE-11157",
    "HIVE-11135", "HIVE-11122", "HIVE-11120", "HIVE-11118", "HIVE-11102", "HIVE-11051", "HIVE-11043", "HIVE-11035", "HIVE-11031", "HIVE-11023", "HIVE-10996", "HIVE-10974",
    "HIVE-10972", "HIVE-10707", "HIVE-10651", "HIVE-10233", "HIVE-10021", "HIVE-9499", "HIVE-4243"
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
                commit["files"] = extract_commit_content(commit["sha"])
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
