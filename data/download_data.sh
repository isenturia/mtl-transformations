#!/bin/bash
# Downloads the three passivization datasets from Karl Mulligan's OSF page
# (https://osf.io/yrq3j/) into this data/ folder.

cd "$(dirname "$0")"

# Side task: None
curl -L -o passive.train https://osf.io/download/aenp3/
curl -L -o passive.dev   https://osf.io/download/wsy3b/
curl -L -o passive.test  https://osf.io/download/2w9th/
curl -L -o passive.gen   https://osf.io/download/t6m8q/

# Side task: Linear (question formation, move the first auxiliary)
curl -L -o passive_question_first.train https://osf.io/download/vae5q/
curl -L -o passive_question_first.dev   https://osf.io/download/jrhzq/
curl -L -o passive_question_first.test  https://osf.io/download/sx5jv/
curl -L -o passive_question_first.gen   https://osf.io/download/by5x6/

# Side task: Hierarchical (question formation, move the main-clause auxiliary)
curl -L -o passive_question_main.train https://osf.io/download/arqmj/
curl -L -o passive_question_main.dev   https://osf.io/download/v32du/
curl -L -o passive_question_main.test  https://osf.io/download/jcs94/
curl -L -o passive_question_main.gen   https://osf.io/download/epvnq/
