"""
An implementation of a constant-distortion two round
quasi-polynomial participatory budgeting voting rule
"""
from itertools import combinations
import csv, os, glob
from functools import lru_cache
from math import log, ceil
from fractions import Fraction
from random import sample, random
from pprint import pformat
import signal, pickle
import logging

# logging.DEBUG to see copeland info and all feasible sequences
# logging.INFO to see all intermediate steps
# logging.WARNING for no logging output
logging.basicConfig(format='%(message)s', level=logging.WARNING)

path = "2024-07-27_23-48-22_pabulib"  # set this variable

def copeland(rankings, full_ranking=False):
    """Given a list of rankings over some bundles, return the copeland winner.
    If full_ranking is True, return the result of applying copeland iteratively.

    Ties add 0 to the scores of the alternatives instead of 1/2 because it
    shouldn't matter, and now the alternatives sorted by the scores correspond
    to copeland being applied iteratively.
    """

    logging.debug(f"\nCopeland:\nRankings:\n{pformat(rankings)}")
    bundles = rankings[0]
    num_bundles = len(bundles)

    # Assumes a full ranking
    assert({len(ranking) for ranking in rankings} == {num_bundles})
    position = {bundle: [] for bundle in bundles}
    for ranking in rankings:
        for i, bundle in enumerate(ranking):
            position[bundle].append(i)

    pairwise_scores = {(b1, b2): 0 for b1 in bundles for b2 in bundles}
    for i in range(num_bundles):
        for j in range(i + 1, num_bundles):
            b1 = bundles[i]
            b2 = bundles[j]
            score = 0
            
            for k in range(len(position[b1])):
                if position[b1][k] < position[b2][k]:
                    pairwise_scores[(b1, b2)] += 1
                else:
                    pairwise_scores[(b2, b1)] += 1
        
    logging.debug(f"Scores:\n{pairwise_scores}")

    scores = {b: 0 for b in bundles}
    for i in range(num_bundles):
        b1 = bundles[i]
        for j in range(num_bundles):
            b2 = bundles[j]
            if pairwise_scores[(b1, b2)] > 0:
                scores[b1] += 1

    if not full_ranking:
        return max(scores, key=scores.get)
    ranking = []
    seen = set()
    while len(ranking) != len(bundles):
        top = max(scores, key=scores.get)
        ranking.append(top)
        seen.add(top)
        for i in range(num_bundles):
            b = bundles[i]
            if b in seen: continue
            if pairwise_scores[(b, top)] > 0:
                scores[b] -= 1
        scores[top] = -1

    logging.debug(ranking)
    return ranking


def build_buckets(alternatives):
    """Place the alternatives into log(m) buckets where the cost of any two alternatives in
    each bucket either differ by at most a factor of 2
    or are both alternatives are worth <= 1/m
    """
    m = len(alternatives)
    num_of_buckets = ceil(log(m, 2)) + 1
    buckets = [set() for _ in range(num_of_buckets)]
    for alternative in alternatives:
        cost = costs[alternative]
        for i in range(num_of_buckets - 1):
            if cost * (2 ** (1 + i)) >= 1:
                buckets[i].add(alternative)
                break
        else:
            buckets[-1].add(alternative)

    return buckets


def create_bundles(ranked_buckets, debugging_mode=False):
    """Given a ranking of the buckets, return a list of premade bundles
    """
    num_of_buckets = len(ranked_buckets)
    
    @lru_cache
    def feasible_sequences(start, running_total):
        """Generate all maximal feasible sequences of powers of 2 or 0
        starting from the start'th number that sums to 1 - running_total.
        """
        if start == num_of_buckets:
            if running_total == 1:
                return [[]]
            return []

        power_of_two = 0
        highest_cost = Fraction(1, 2 ** start) # An upper bound on the cost in the bucket

        return_list = []
        while running_total + power_of_two * highest_cost <= 1:
            for feasible_sequence in feasible_sequences(start + 1, running_total + power_of_two * highest_cost):
                return_list.append([power_of_two] + feasible_sequence)
            power_of_two = 1 if power_of_two == 0 else power_of_two * 2
        return return_list

    bundles = set()
    for feasible_sequence in feasible_sequences(0, 0):
        logging.debug(feasible_sequence)
        bundle = []
        for i, ranked_bucket in enumerate(ranked_buckets):
            num_to_take = feasible_sequence[i]
            bundle.extend(ranked_bucket[:num_to_take])
        if len(bundle) != 0 and not tuple(bundle) in bundles:
            bundles.add(tuple(bundle))

    remove = set()
    for bundle in bundles:
         if bundle in remove: continue
         for compare in bundles:
             if compare in remove or compare == bundle: continue
             if set(bundle).issubset(set(compare)):
                 remove.add(bundle)
                 break
   
    bundles = bundles - remove
    return bundles


def preprocess_bundles(bundles, copeland_rankings, debugging_mode=False):
    """Remove a bundle if it is strictly worse by the
    Copeland ordering."""
    remove = set()
    score = {alternative: -i for i, alternative in enumerate(copeland_rankings)}
    def compare_bundles(bundle, comparison):
        if len(bundle) > len(comparison): return False
        for i in range(len(bundle)):
            if score[bundle[i]] > score[comparison[i]]: return False
        return True

    for bundle in bundles:
        if bundle in remove: continue
        for comparison in bundles:
            if bundle == comparison or comparison in remove: continue

            sorted_bundle = sorted(bundle, key=score.get, reverse=True)
            sorted_comparison = sorted(comparison, key=score.get, reverse=True)

            if compare_bundles(sorted_bundle, sorted_comparison):
                remove.add(bundle)
                break

    return bundles - remove, len(remove)


def rank(bundles, i):
    """Return voter i's preferences over the bundles.
    Random for now.
    """
    return sample(bundles, len(bundles))


def fill_in_blanks(vote, alternatives_set):
    unranked = list(alternatives_set - set(vote))
    rest = sample(unranked, len(unranked))
    return vote + rest


def analysis(costs, voters, prune, find_winning_bundle, debugging_mode=False):
    alternatives = costs.keys()
    m = len(alternatives)
    n = len(voters)
    logging.info(f"Alternatives:{pformat(costs)}")

    alternatives_set = set(alternatives)
    for i, vote in enumerate(voters):
        voters[i] = fill_in_blanks(vote, alternatives_set)

    ### Round 1
    # Elicitate from each voter a full ranking over the alternatives

    # With random preferences
    #voters = [
    #    rank(alternatives, i)
    #    for i in range(n)
    #]

    if prune or find_winning_bundle:
        iterated_copeland = copeland(voters, True)
    else:
        # For efficiency, just use any ordering of the bundles
        # to find number of bundles used
        iterated_copeland = voters[0]

    # Build the bundles
    buckets = build_buckets(alternatives)
    logging.info(f"\nBuckets (ith bucket contains all alternatives of cost (2^{{-i}}, 2^{{-i-1}}]):\n{pformat(buckets)}")

    ranked_buckets = list(map(
        lambda bucket: sorted(bucket, key=iterated_copeland.index),
        buckets
    ))
    logging.info(f"\nRanked Buckets:{pformat(ranked_buckets)}")

    bundles = create_bundles(ranked_buckets, debugging_mode)
    old_bundles = bundles
    if prune:
        bundles, len_removed = preprocess_bundles(bundles, iterated_copeland, debugging_mode)
        if len(bundles) == 1 and debugging_mode:
            breakpoint()
    logging.info(f"\nBundles:\n{pformat(bundles)}")

    if not find_winning_bundle:
        if prune:
            return len(bundles), len_removed
        return len(bundles), 0

    ### Round 2
    # Elicitate from each voter a full ranking over the premade bundles
    ranking_over_bundles = [
        rank(bundles, i)
        for i in range(n)
    ]

    winning_bundle = copeland(ranking_over_bundles)
    logging.info(f"""
    Number of alternatives: {m}
    Number of bundles: {len(bundles)}
    Winning bundle: {winning_bundle}
    """)

    if prune:
        return winning_bundle, len(bundles), len_removed
    return winning_bundle, len(bundles)


pb_files = glob.glob(os.path.join(path, '*.pb'))
data_path = 'data.pkl'
repeat = 3
prune = False
debugging_mode = False

try:
    with open(data_path, 'rb') as data_file:
        data = pickle.load(data_file)
except FileNotFoundError:
    logging.warning(f"File {data_path} doesn't exist")
    data = {}

if data_path == "debug.pkl":
    logging.info("Debugging mode is set!")
    data = {}
    debugging_mode = True
    

def dump_data():
    with open(data_path, 'wb') as data_file:
        pickle.dump(data, data_file)

def end_program(signal_number=None, frame=None):
    dump_data()
    exit()

signal.signal(signal.SIGINT, end_program)

for pb_file in pb_files:
    if pb_file in data: continue
    logging.warning(pb_file)
    with open(pb_file, 'r', newline='', encoding="utf-8") as csvfile:
        meta = {}
        projects = {}
        votes = {}
        section = ""
        header = []
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            if len(row) == 0: continue
            if str(row[0]).strip().lower() in ["meta", "projects", "votes"]:
                section = str(row[0]).strip().lower()
                header = next(reader)
            elif section == "meta":
                meta[row[0]] = row[1].strip()
            elif section == "projects":
                projects[row[0]] = {}
                for it, key in enumerate(header[1:]):
                    projects[row[0]][key.strip()] = row[it+1].strip()
            elif section == "votes":
                votes[row[0]] = {}
                for it, key in enumerate(header[1:]):
                    votes[row[0]][key.strip()] = row[it+1].strip()

    budget = float(meta['budget'])
    costs = {project : float(projects[project]['cost']) / budget for project in projects}

    def reformat(vote, shuffle=False):
        if ',' in vote:
            vote = vote.split(',')
            vote = list(dict.fromkeys(vote))
            if shuffle:
                return sample(vote, len(vote))
            return vote
        return [vote]

    average_num_bundles, average_len_removed = 0, 0
    for _ in range(repeat):
        shuffle_votes = meta['vote_type'] == 'approval'
        voters = [reformat(votes[vote]['vote'], shuffle_votes) for vote in votes]

        num_alternatives = len(costs)
        num_bundles, *len_removed = analysis(costs, voters, prune, False, debugging_mode)
        average_num_bundles += num_bundles
        if prune: average_len_removed += len_removed[0]
    average_num_bundles /= repeat
    if prune:
        average_len_removed /= repeat
        data[pb_file] = (num_alternatives, average_num_bundles, average_len_removed)
        logging.warning(f"{num_alternatives}, {average_num_bundles}, {average_len_removed}\n")
    else:
        data[pb_file] = (num_alternatives, average_num_bundles)
        logging.warning(f"{num_alternatives}, {num_bundles}\n")
    dump_data()

