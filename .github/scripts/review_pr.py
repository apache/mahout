#!/usr/bin/env python3
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import sys
from github import Github
from gofannon.github.pr_review_tool import PRReviewTool


def check_env_vars():
    required_vars = [
        "GITHUB_TOKEN",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL_NAME",
        "PR_NUMBER",
        "REPO_NAME",
    ]
    for var in required_vars:
        if not os.environ.get(var):
            sys.exit(f"Error: Required environment variable '{var}' is missing.")


check_env_vars()


def main():
    pr_number = int(os.environ["PR_NUMBER"])
    repo_name = os.environ["REPO_NAME"]
    pr_review_tool = PRReviewTool()
    review_summary = pr_review_tool.fn(pr_number=pr_number, repo_name=repo_name)

    # Post the review comment to the pull request using PyGithub.
    g = Github(os.environ["GITHUB_TOKEN"])
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    pr.create_issue_comment(review_summary)


if __name__ == "__main__":
    main()
