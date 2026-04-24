# Crossover Effectiveness in QCS

This repository contains the source code to the paper
_"Revisiting Crossover Effectiveness in Quantum Circuit Synthesis"_
by Stein, Klikovits, and Wimmer from the [Institute of Business Informatics - Software Engineering](https://se.jku.at/) at the [Johannes Kepler University](https://www.jku.at/en), Linz.

## Setup

To run the code in this repository locally,
create a [virtual environment](https://docs.python.org/3/library/venv.html) and
install the required python dependencies using

```
pip install -r requirements.txt
```

Alternatively, you can use the provided `docker-compose.yml`
file to run the experiments within a docker container.
Make sure to adjust the _command_ field in the docker compose
file in accordance with the experiment configurations you want
to run.

## Run and Evaluate

The entry point of the crossover effectiveness experiments is the `main.py` file.
This script has multiple parameters which you can tweak in accordance
with your needs.
For a description of the required and optional arguments of this script
as well as their accepted parameter sets,
run

```
python main.py --help
```

For each experiment, the resulting GA performance and experiment meta
data are written to the `results/` directory.

The `evaluation.ipynb` file then reads from this folder and creates
figures in accordance with the paper.
You might have to adjust some paths in this file depending on how you
group the experiment runs in the `results/` folder.

If you have any questions, feel free to reach out!

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
