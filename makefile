run_simulation:
	python3 -m src.simmulation.automatic

run_game:
	python3 -m src.simmulation.interactive

generate_data:
	python3 -m src.gen_maze_data.main

generate_dataset:
	python3 -m src.gen_dataset.main

install:
	python3 -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt

activate:
	source venv/bin/activate
