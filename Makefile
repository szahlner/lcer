# Mostly taken from stablebaselines3 at: https://github.com/DLR-RM/stable-baselines3/blob/master/Makefile

SHELL=/bin/bash
LINT_PATHS=lcer/ tests/

pytest:
	# Run tests
	./scripts/run_tests.sh

lint:
	# Stop the build if there are Python syntax errors or undefined names
	# See https://lintlyci.github.io/Flake8Rules/
	flake8 ${LINT_PATHS} --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings.
	flake8 ${LINT_PATHS} --count --exit-zero --statistics

format:
	# Sort imports
	isort --profile black ${LINT_PATHS}
	# Reformat using black
	black -l 127 ${LINT_PATHS}

check-codestyle:
	# Sort imports
	isort --check --profile black ${LINT_PATHS}
	# Reformat using black
	black --check -l 127 ${LINT_PATHS}

# Build docker images
# If you do export RELEASE=True, it will also push them
docker: docker-cpu docker-gpu

docker-cpu:
	./scripts/build_docker.sh

docker-gpu:
	USE_GPU=True ./scripts/build_docker.sh