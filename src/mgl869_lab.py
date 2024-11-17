from pathlib import Path
import csv
import os
import json
from modules.jira_extractor import extract_issues
from modules.commit_extractor_localDirectory import get_commits

file_path = Path('modules/ReleaseVersion_Commit.json')


def load_release_versions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def write_commits_to_csv(commit_list, output_file):
    # Check if the file exists
    if os.path.exists(output_file):
        # Delete the file if it exists
        os.remove(output_file)
        print(f"Deleted existing file: {output_file}")

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['key', 'filename'])

        for commit in commit_list:
            key = commit['key']
            for bug_file in commit['files']:
                writer.writerow([key, bug_file])


release_versions = load_release_versions(file_path)

for version in release_versions:
    JIRA_SEARCH_FILTER = "project = HIVE AND issuetype = Bug AND status in (Resolved, Closed) AND resolution = Fixed AND affectedVersion = " + version
    OUTPUT_FILE_NAME = "Bugs_" + version + ".csv"
    OUTPUT_FILE = Path(OUTPUT_FILE_NAME)  # Extraire les commits dans ce fichier

    print("Extracting jira issues...")
    issues = {}
    for issue in extract_issues(JIRA_SEARCH_FILTER, ["versions"]):
        issues[issue["key"]] = {"affectedVersions": [e["name"] for e in issue["fields"]["versions"]]}
    print(f"\tFound {len(issues)} issues.\n")

    # Note: Hive doit être sur la main branch pour que le code bonne les bons résultats
    print("Extracting github commits...")
    repo_path = r'C:\Users\lafor\Desktop\ETS - Cours\MGL869-01_Sujets speciaux\Laboratoire\Hive\hive'
    print(repo_path)
    commits_list = get_commits(repo_path, issues)

    write_commits_to_csv(commits_list, OUTPUT_FILE)

    del commits_list
    del issues
