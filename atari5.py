import numpy as np
import pandas
import pandas as pd
import itertools
import sklearn
import sklearn.linear_model
import statsmodels
import statsmodels.api as sm
import json
import csv
import matplotlib.pyplot as plt
import multiprocessing
import functools
import time
from sklearn.model_selection import cross_val_score
from tqdm.contrib.concurrent import process_map  # or thread_map
from tqdm import tqdm

from atari_util import printable_57, canonical_57

# not using an intercept means input scores of 0 -> output scores 0.
USE_INTERCEPT = False

# adds a fix to the riverraid results in the rainbow paper...
USE_RAINBOW_RIVERRAID = True # should be true
USE_UPDATED_A3C = False # Atari-5 paper used scores 'from papers with code' which excludes pheonix,
                        # including the pheonix changes coefficently only very slightly.

# First return used different settings so is not included, but sometimes I need to load it so that I can evaluate
# performance on it.
INCLUDE_FIRSTRETURN = False

# number of CPU workers to use
PROCESSES = 12

BANNED_ALGOS = {
    # these algorithms are not included in the dataset. Reason is given for each one.
    "PapersWithCode.Go-Explore",    # used 400k max_frames
    "PapersWithCode.UCT",           # use of emulator as model
    "PapersWithCode.Rainbow",       # incomplete data, but added via RAINBOW.csv
    "PapersWithCode.DQN noop",      # this is fine, but I use the version from RAINBOW
    "PapersWithCode.Duel noop",     # this is fine, but I use the version from RAINBOW
    "PapersWithCode.DDQN (tuned) noop", # this is fine, but I use the version from RAINBOW
    "SEED.R2D2 Ref",                # remove (matched) duplicate
    "PapersWithCode.Advantage Learning", # using alternative
    "PapersWithCode.Persistent AL", # using alternative
    "PapersWithCode.C51", # using alternative
    "PapersWithCode.C51 noop", # using alternative
    "PapersWithCode.Bootstrapped DQN", # using alternative
    "PapersWithCode.QR-DQN-1", # using alternative
}

game_map = {
        "montezumasrevenge": "montezumarevenge",
        'upanddown': 'upndown',
        # these games are ignored
        'pooyan': None,
        'journeyescape': None,
        'elevatoraction': None,
        'carnival': None,
        'airraid': None,
    }

# global vars
algo_scores = None

# also allow just the first word in a multi word game
for game in printable_57:
    if " " in game:
        game_map[game.split(" ")[0].lower()] = "".join(game.split(' ')).lower()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Algorithm():
    def __init__(self, name: str, source: str, paper_title: str, paper_year: int):
        self.name = name
        self.source = source
        self.paper = paper_title
        self.year = paper_year
        self.game_scores = {}
        self.log_target = None

    @property
    def full_name(self):
        return self.source+'.'+self.name

    def __str__(self):
        return self.full_name

    @property
    def a57_games(self):
        return {k: v for k, v in self.game_scores.items() if k in canonical_57}


def games_matched(s1, s2):
    """
    Checks if two sets of game scores are similar to each other.
    s1, and s2 are dictionaries mapping from game name to score.
    """
    result = 0
    for game, score in s1.items():
        if game in s2 and abs(s2[game] - score) / (abs(score)+1e-3) < 0.01:
            result += 1
    return result


def transform(x):
    return np.log10(1+np.clip(x, 0, float('inf')))


def inv_transform(x):
    return (10 ** x) - 1

def clean_up_atari_game_name(name, keep_games=False):
    """
    Takes the name of an atari game and returns it's normalized form
    """

    name = "".join([c for c in name if c.isalpha()])
    name = name.lower()
    if name in game_map:
        result = game_map[name]
        if result is None and keep_games:
            return name
        else:
            return result
    else:
        return name


def convert_atari_data(path, output_file, algorithm_name_map = None):
    """
    Extract data from json file (source from https://github.com/paperswithcode/paperswithcode-data)
    Path is to JSON file from paperswithcode
    Saves a CSV file containing results
    """

    with open(path, 'r') as t:
        data = json.load(t)

    atari_key = [i for i in range(len(data)) if 'Atari' in data[i]['task']]
    assert len(atari_key) == 1  # usually item 407, but this might change...
    atari_datasets = data[atari_key[0]]["datasets"]

    games = [atari_datasets[i]['dataset'] for i in range(len(atari_datasets))]
    print(f'Found {len(games)} games')

    def sanitize(s: str):
        return "".join(c for c in s if c not in [','])

    atari_prefix = 'atari 2600 '

    with open(output_file, 'w', newline='', encoding='utf-8') as t:
        csv_writer = csv.writer(t, delimiter=',')
        csv_writer.writerow(['Algorithm', 'Score', 'Extra Training Data', 'Paper Title', 'Date', 'Game'])

        for dataset in atari_datasets:

            for row in dataset['sota']['rows']:
                if 'Score' not in row['metrics']:
                    continue
                algorithm = row['model_name']

                if algorithm_name_map is not None and algorithm in algorithm_name_map:
                    algorithm = algorithm_name_map[algorithm]

                score = sanitize(row['metrics']['Score'])
                extra_training_data = row['uses_additional_data']  # for now

                paper_title = row['paper_title']
                paper_date = row['paper_date']
                if paper_date is not None:
                    paper_date = paper_date[:4] # year is enough
                game = dataset['dataset'].lower()
                if game.startswith(atari_prefix):
                    game = game[len(atari_prefix):]
                game = clean_up_atari_game_name(game)
                if game is None:
                    continue
                if game not in canonical_57:
                    print(f" -game {game} ignored")
                    continue
                csv_writer.writerow([algorithm, score, extra_training_data, paper_title, paper_date, game])


def count_57_games(algo):
    """ Returns number of games within the canonical set of 57 that
    this algorithmh has scores for. """
    return 57-len(missing_57_games(algo))


def excess_57_games(algo):
    """ Returns number of games within the canonical set of 57 that
    this algorithmh has scores for. """
    subset = algo_scores[algo_scores["Algorithm"] == algo]
    # filter by 57 games
    subset = subset[np.logical_not(subset["In57"])]
    return subset


def missing_57_games(algo):
    """ Returns all games missing from the Atari-57 datset. """
    subset = algo_scores[algo_scores["Algorithm"] == algo]
    return [game for game in canonical_57 if game not in list(subset["Game"])]


def calculate_median(algo, scores):
    """ Calculate the median score for a given algorithm. """

    # filter by algorithm
    subset = scores[scores["Algorithm"] == algo]

    if len(subset) == 0:
        return float("nan")

    return np.median(subset["Normalized"])


def get_subset(games_set, scores=None):
    """ Returns rows matching any algorithm in games_set"""
    scores = scores if scores is not None else algo_scores
    return scores[[game in games_set for game in scores["Game"]]]

def fit_model(games_subset, algo_scores, intercept=False):
    """
    Fit a linear regression model to games subset.

    @algo_scores: dataframe containing scores for each algorithm / environment,
    @returns:
        lm, (X, y), algos
        lm is the linear model
        X are the input used to fit model
        y are the targets used to fit model

    """
    # scores = get_subset(games_subset, algo_scores)
    # scores = scores[scores["train"]]
    #
    # X_all = scores.pivot_table(
    #     index='Algorithm',
    #     columns="Game",
    #     values="LogNormalized",
    #     fill_value=None,
    # )[list(games_subset)]

    X_all = x_pivot[list(games_subset)]

    # switch to numpy (because I'm paranoid about pandas and removing rows...)

    X = np.asarray(X_all)
    y = np.asarray([algorithms[algo].log_target for algo in X_all.index])
    algorithm_names = np.asarray(X_all.index)
    mask = np.all(np.logical_not(np.isnan(X)), axis=1)
    X = X[mask]
    y = y[mask]
    algorithm_names = algorithm_names[mask]

    X = pandas.DataFrame(X, index=algorithm_names, columns=games_subset)

    if len(X) < 20:
        print(f"Warning! Too many missing samples for: {games_subset}")

    lm = sklearn.linear_model.LinearRegression(fit_intercept=intercept)
    lm.fit(X, y)

    return lm, (X, y)


class RegressionEvaluation:

    def __init__(self, games_subset, intercept=False):
        """
        True scores must be in sorted order according to algos

        @param k_fold_threshold: If given only models with an MSE less than this value will have cvs calculated.
            This can speed things up a lot, by excluding obviously bad models from the fairly slow K-fold evaluation.
        """

        lm, (X, log_y) = fit_model(games_subset, algo_scores, intercept=intercept)

        self.intercept = intercept
        self.games_subset = games_subset
        self.N = len(X)

        log_predicted_scores = lm.predict(X)

        self.log_errors = log_y - log_predicted_scores
        self.log_mse = (self.log_errors ** 2).mean()
        self.log_mae = np.abs(self.log_errors).mean()
        self.log_tss = np.var(log_y, ddof=0) * self.N
        self.r2 = 1 - (self.log_mse * self.N) / self.log_tss

        # cross validation score is calculated later
        self._cross_validation_mse = None
        self._cross_validation_mae = None
        self._coef = [lm.intercept_] + [lm.coef_]


    def _get_cross_validation_score(self, squared=True):
        # mean of means, shouldn't really do this, but bins will be either 6 or 7, so shouldn't
        # make much difference.
        lm, (X, y_raw) = fit_model(self.games_subset, algo_scores, intercept=self.intercept)
        return -np.mean(cross_val_score(
            lm, X, y_raw, cv=sklearn.model_selection.KFold(10, shuffle=True, random_state=1982),
            scoring='neg_mean_squared_error' if squared else 'neg_mean_absolute_error'
        ))

    @property
    def cv_mse(self):
        # lazy cross validation score
        if self._cross_validation_mse is None:
            self._cross_validation_mse = self._get_cross_validation_score()
        return self._cross_validation_mse

    @property
    def cv_mae(self):
        # lazy cross validation score
        if self._cross_validation_mae is None:
            self._cross_validation_mae = self._get_cross_validation_score(squared=False)
        return self._cross_validation_mae

    @property
    def cv_r2(self):
        # lazy cross validation score
        return 1 - (self.cv_mse * self.N) / self.log_tss

    def __str__(self):
        return f"{self.games_subset}={self.cv_mse ** 0.5:.3f}"


def search_regression(r=3, always_envs=None, banned_games=None, verbose=True, game_set=None, top_k=57, intercept=False):
    """
    Search over subsets.
    """

    if game_set is None:
        game_set = canonical_57

    if banned_games is None:
        banned_games = []
    else:
        banned_games = list(banned_games)

    if always_envs is None:
        always_envs = tuple()

    counter = 0
    print("Checking all sets of {} games.".format(r))

    # these games are always banned, as they do not have enough datapoints.
    games_to_search_through = [game for game in game_set if game.lower() not in banned_games and game.lower() not in always_envs]

    combinations = list(itertools.combinations(games_to_search_through, r-len(always_envs)))

    # add always envs in...
    combinations = [x+always_envs for x in combinations]

    start_time = time.time()

    args = {
        'intercept': intercept,
    }

    if PROCESSES > 1:
        results = process_map(
            functools.partial(RegressionEvaluation, **args),
            combinations,
            max_workers=PROCESSES,
            chunksize=1024
        )
        results = list(results)
    else:
        results = []
        for c in tqdm(combinations):
            results.append(RegressionEvaluation(c, **args))

    time_taken = time.time() - start_time
    fps = len(results) / time_taken

    # print(f"Generated {len(results)} models in {time_taken:.1f}s at {fps:.1f} models/second.")

    # we take the top_k in terms of log_mse, then calculate validation scores for those, and sort by validation.
    # Due to the small number of parameters the ordering generated from the cross validation scores and log_mse scores
    # match very closely, and we only care about the top one anyway.
    results.sort(reverse=True, key=lambda x: x.log_mse)
    results = results[-top_k:]
    results.sort(reverse=True, key=lambda x: x.cv_mse)

    print(f"{'subset':<60} {'rmse':<10} {'cv_rmse':<10} {'coef'}")

    for re in results:
        if verbose:
            print(f"{str(re.games_subset):<60} {re.log_mse**0.5:<10.3f} {re.cv_mse ** 0.5:<10.3f} {re._coef}")
        counter += 1

    time.sleep(0.100) # wait for TQDM process to die.
    print()

    return results


def bold(x):
    return f"{bcolors.BOLD}{x}{bcolors.ENDC}"


def run_init(do_not_train_on=None, verbose=False):
    """
    Run initialization.
    sets the following global varaibles
        all_algo_scores: same as algo_scores but includes algorithms that do not meet criteria.
        algo_scores: dataframe containing where each row is a score for an algorithm on a game (with bad algorithms filtered out)
        algorithms: dictionary of Algorithms instances with useful stats for each algorithm.

    """

    global all_algo_scores
    global algo_scores
    global x_pivot # pivot table version
    global algorithms

    do_not_train_on = do_not_train_on or []

    if verbose:
        print("Number of canonical games {}".format(len(canonical_57)))

    pd.set_option('display.max_columns', None)

    # load human and random results
    human_scores = pd.read_csv("Atari-Human.csv", dtype={"Score": float})

    # load in the paper with code data (this is in a slightly different format

    pwc_file = "PapersWithCode.csv"
    convert_atari_data('evaluation-tables (Feb22).json', pwc_file)
    algo_scores = pd.read_csv(pwc_file, dtype={"Score": float})
    algo_scores['Source'] = pwc_file.split('.')[0]

    def load_data(filename):
        """
        Expected format is Game, <algo1>, <algo2>, ...
        """
        new_data = pd.read_csv(filename)

        # standardise names
        new_data['Game'] = [clean_up_atari_game_name(x) for x in new_data['Game']]

        # covert from wide to long
        new_data = pd.melt(
            new_data,
            id_vars=new_data.columns[0:1],
            value_vars=new_data.columns[1:],
            value_name='Score',
            var_name='Algorithm'
        )

        # change '-' to none
        for idx, row in new_data.iterrows():
            if row['Score'] == "-":
                row['Score'] = None
            else:
                # force float type
                # silly NGU...
                if type(row['Score']) is str and row['Score'] != '' and row['Score'][-1] == 'k':
                    row['Score'] = float(row['Score'][:-1]) * 1000
                else:
                    row['Score'] = float(row['Score'])

        # remove invalid games
        new_data = new_data.dropna()
        new_data = new_data.copy()

        # add source
        new_data['Source'] = filename.split('.')[0]
        new_data['Score'] = new_data['Score'].astype('float')

        return new_data
        # append data

    ADDITIONAL_DATA_SETS = [
        'RAINBOW (fixed).csv' if USE_RAINBOW_RIVERRAID else 'RAINBOW.csv',
        'SEED.csv',
        'INCREASE.csv',
        'DRLQR.csv',
        'BOOTDQN.csv',
        'LASER.csv',
        'REVISITING.csv',
        'NEAT.csv',
        'REACTOR.csv',
        'NGU.csv',
        'SIMPLE.csv',
    ]

    if USE_UPDATED_A3C:
        # use the scores from the paper, rather than from paperswithcode, as PWC misses some games.
        ADDITIONAL_DATA_SETS.append('A3C.csv')
        global BANNED_ALGOS
        BANNED_ALGOS = BANNED_ALGOS.union([
            "PapersWithCode.A3C FF hs",
            "PapersWithCode.A3C LSTM hs",
            "PapersWithCode.A3C FF (1 day) hs",
        ])

    if INCLUDE_FIRSTRETURN:
        ADDITIONAL_DATA_SETS.append("FIRSTRETURN.csv")

    # load in data from each paper
    new_data = [
        load_data(x) for x in ADDITIONAL_DATA_SETS
    ]

    algo_scores = pd.concat([algo_scores, *new_data])

    # make all names lower case
    human_scores['Game'] = [clean_up_atari_game_name(x) for x in human_scores['Game']]
    algo_scores["Game"] = algo_scores["Game"].str.lower()

    # remove data from banned algorithms
    algo_scores.reset_index(drop=True, inplace=True) # needed as indexes are duplicated...
    drop_rows = []
    keep_rows = []
    for idx, row in algo_scores.iterrows():
        full_name = row["Source"] + "." + row["Algorithm"]
        if full_name in BANNED_ALGOS or row["Algorithm"][0] == "_":
            drop_rows.append(idx)
        else:
            keep_rows.append(idx)
    algo_scores.drop(drop_rows, axis=0, inplace=True)
    algo_scores.reset_index(drop=True, inplace=True)
    if len(drop_rows) > 0:
        print(f"Dropped {len(drop_rows)} rows.")

    for index, row in algo_scores.iterrows():
        if row["Game"] not in canonical_57:
            print(f"Invalid game '{row['Game']}' on algorithm {row['Algorithm']}")

    algo_scores = algo_scores.merge(human_scores[["Game", "Random", "Human"]], on="Game", how="left")
    algo_scores["Normalized"] = 100 * (algo_scores["Score"] - algo_scores["Random"]) / (
            algo_scores["Human"] - algo_scores["Random"])
    algo_scores["LogNormalized"] = transform(algo_scores["Normalized"])

    algo_scores["In57"] = [game in canonical_57 for game in algo_scores["Game"]]

    all_algorithms_list = set(algo_scores["Algorithm"])
    if verbose:
        print("All algorithms:", all_algorithms_list)

    for game in do_not_train_on:
        assert game in all_algorithms_list, f"{game} missing from algorithms list"

    algo_scores["train"] = [game not in do_not_train_on for game in algo_scores["Algorithm"]]

    # get algorithm stats and look for (algorithm,score) pair duplications
    algorithms = {}
    for index, row in algo_scores.iterrows():

        name = row['Algorithm']
        game = row['Game']

        norm_score = 100 * (row["Score"] - row["Random"]) / (row["Human"] - row["Random"])

        source = row['Source']

        paper_title = row['Paper Title']
        if type(paper_title) is not str:
            paper_title = "?"

        try:
            paper_year = int(row['Date'])
        except:
            paper_year = 0

        if name in algorithms:
            # just verify
            algo = algorithms[name]
            if game in algo.game_scores:
                print(f"Warning, found duplicate entries for pair ({name}, {game}) in {source}")
            if source != algo.source:
                print(f"Warning, found algorithm {name} in multiple sources. {algo.source}, {source}")
        else:
            algo = Algorithm(name=name, source=source, paper_title=paper_title, paper_year=paper_year)

        if norm_score is not float('nan'):
            algo.game_scores[game] = norm_score
        else:
            print(f"Warning {algo}:{game} has nan score.")

        algorithms[name] = algo

    for algo in algorithms.values():
        algo.a57_median = np.median(list(algo.a57_games.values()))
        algo.log_target = transform(algo.a57_median)
        algo.is_good = len(algo.a57_games.values()) >= 40 and algo.a57_median > 40 and algo.full_name not in BANNED_ALGOS
    good_algos = [algo.name for algo in algorithms.values() if algo.is_good]
    all_algorithms = [algo.name for algo in algorithms.values()]
    print(f"Found {bcolors.BOLD}{len(all_algorithms)}{bcolors.ENDC} algorithms, with {bcolors.WARNING}{len(good_algos)}{bcolors.ENDC} meeting requirements.")

    # stub:
    print(sorted(good_algos))

    # make sure we don't have duplicates (again...)
    for a1_name in algorithms.keys():
        for a2_name in algorithms.keys():
            if a1_name == a2_name:
                continue
            a1 = algorithms[a1_name]
            a2 = algorithms[a2_name]
            matches = games_matched(a1.a57_games, a2.a57_games)
            if matches > 25:
                print(f"Algorithms {bold(a1)} and {bold(a2)} look similar, matched on {matches}/{len(algorithms[a1_name].a57_games)} of games.")

    # median before algorithm filter...
    all_median_scores = {
        k: calculate_median(k, get_subset(canonical_57)) for k in all_algorithms
    }

    # filter bad algorithms
    algo_scores["good"] = [algo in good_algos for algo in algo_scores["Algorithm"]]
    algo_scores = algo_scores[algo_scores['good']]
    algo_scores["Game"] = algo_scores["Game"].astype('category')  # faster?
    median_sorted_scores = sorted([(v, k) for k, v in all_median_scores.items()])

    # filtered results, and calculate true targets
    all_algo_scores = algo_scores.copy()
    algo_scores = algo_scores[[algo in good_algos for algo in algo_scores["Algorithm"]]]
    algo_scores['LogNormalized'] = transform(algo_scores['Normalized'])

    # pregenerate the pivot table
    scores = algo_scores[algo_scores["train"]]
    x_pivot = scores.pivot_table(
        index='Algorithm',
        columns="Game",
        values="LogNormalized",
        fill_value=None,
    )


    if verbose:
        print(f"Found {len(good_algos)} datapoints with 40 or more games.")
        for n_games in reversed(range(1, 57 + 1)):
            matching = [algo for algo in all_algorithms if count_57_games(algo) == n_games]
            if len(matching) > 0:
                print(f"[{n_games:02d}] {matching}")

        print()
        print("Missing games:")
        for algo in good_algos:
            print(f" -{algo}: {missing_57_games(algo)}")


        print()
        print("Median_57 scores:")

        for score, algo in median_sorted_scores:
            if algo not in good_algos:
                continue
            marker = "*" if algo in do_not_train_on else ""
            if algo not in good_algos:
                marker = marker + " -"
            print(f" -{algo}: {score:.0f} {marker}")


if __name__ == "__main__":

    # ---------------------------------------------------
    # find a good subsets...

    run_init()

    algo_scores.to_csv('dataset.csv')

    args = {
        'top_k': 20,
        'intercept': USE_INTERCEPT,
        #'k_fold_threshold': 0.15, # makes things a bit faster
        #'k_fold_threshold': (3.5**2) # makes things a bit faster
    }

    results = {}

    # games with 50+/62
    #not_enough_data_games = ["defender", "phoenix", "pitfall", "skiing", "solaris", "surround", "yarsrevenge"]
    # games with 40+/62
    not_enough_data_games = ["surround"]

    print("-" * 60)
    print(" Test")
    print("-" * 60)


    results['Atari_Single'] = search_regression(1, intercept=True, top_k=10)[-1]

    args['banned_games'] = not_enough_data_games
    results['Atari_5'] = atari5 = search_regression(5, **args)[-1]
    results['Atari_3'] = atari3 = search_regression(3, game_set=atari5.games_subset, **args)[-1]
    results['Atari_1'] = atari1 = search_regression(1, game_set=atari3.games_subset, **args)[-1]

    print("-" * 60)
    print(" Validation")
    print("-" * 60)

    args['banned_games'] = not_enough_data_games + list(atari5.games_subset)

    # atari3_val is done first as three games is usually enough for validation.
    results['Atari_3_Val'] = atari3_val = search_regression(3, **args)[-1]
    results['Atari_5_Val'] = atari5_val = search_regression(5, always_envs=atari3_val.games_subset, **args)[-1]
    results['Atari_1_Val'] = atari1_val = search_regression(1, game_set=atari3_val.games_subset, **args)[-1]

    print("-" * 60)
    print(" Atari-10")
    print("-" * 60)

    # for when you really need a precise result
    args['banned_games'] = not_enough_data_games + list(atari5_val.games_subset)
    results['Atari_7'] = search_regression(7, always_envs=atari5.games_subset, **args)[-1]
    results['Atari_10'] = search_regression(10, always_envs=atari5.games_subset, **args)[-1]

    # the overlapped version uses validation set games, This is just to see how much more we gain using 10 environments
    # rather than 5.
    # args['banned_games'] = not_enough_data_games
    # results['Atari_10_Overlap'] = atari10_overlap = search_regression(10, always_envs=atari5.games_subset, **args)[-1]

    # show atari_score normalizer settings
    print("SUBSETS = {")
    for k, v in results.items():
        relative_error = v.cv_mae * np.log(10) * 100
        print(f"'{k}': ({list(v.games_subset)}, {list(v._coef[1])}, {relative_error:.1f}),")
    print("}")