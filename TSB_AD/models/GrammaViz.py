"""
This function is adapted from [grammarviz2_src] by [GrammarViz2]
Original source: [https://github.com/GrammarViz2/grammarviz2_src]
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class GrammaViz():
    def __init__(self, window_size=12, word_size=4, alphabet_size=4):
        self.window_size = window_size
        self.word_size = word_size
        self.alphabet_size = alphabet_size
        self.digrams = {}
        self.numRules = 0

    def fit(self, X, y=None):

        self.data = X
        self.length = len(self.data)

        # Z-normalization
        self.z_normalized_ts = self.z_normalize(self.data)

        # Extract subsequences
        self.subsequences, self.sliding_window_indices = self.sliding_window(self.z_normalized_ts, self.window_size)

        # Apply SAX to subsequences
        self.sax_words = self.sax_transform(self.subsequences, self.word_size, self.alphabet_size)

        # Apply Numerosity Reduction
        self.reduced_sax_words = self.numerosity_reduction(self.sax_words)

        # Apply the updated Sequitur to reduced SAX words
        self.sequitur_output = self.run_sequitur(self.reduced_sax_words)

        # Create occurrences array
        self.occurrences = np.zeros_like(self.z_normalized_ts)

        # Stack both arrays together as columns
        self.decision_scores_ = np.column_stack((self.z_normalized_ts, self.occurrences))

        # Map SAX words to original subsequences
        self.mapped_subsequences = self.map_sax_to_subsequences(self.reduced_sax_words, self.z_normalized_ts)

        # Count occurrences
        self.count_occurrences()

        # Find discord
        self.find_discord()


    def generate_time_series_with_anomaly(self):
        time_series = np.sin(np.linspace(0, 4 * np.pi, self.length)) + 0.1 * np.random.randn(self.length)
        for position in self.anomaly_positions:
            time_series[position] += self.anomaly_value  # Adding significant outliers
        return time_series


    def z_normalize(self, time_series):
        scaler = StandardScaler()
        return scaler.fit_transform(time_series.reshape(-1, 1)).flatten()


    def sliding_window(self, time_series, window_size):
        subsequences = []
        indices = []
        for i in range(len(time_series) - window_size + 1):
            subsequence = time_series[i:i + window_size]
            subsequences.append(subsequence)
            indices.append(i)
        return np.array(subsequences), indices


    def sax_transform(self, subsequences, word_size, alphabet_size):
        breakpoints = np.linspace(-1, 1, alphabet_size - 1)
        sax_words = []


        for subsequence in subsequences:
            segment_size = len(subsequence) // word_size
            paa_representation = [np.mean(subsequence[i:i + segment_size]) for i in range(0, len(subsequence), segment_size)]
            sax_word = ''.join([chr(97 + np.sum(mean_value > breakpoints)) for mean_value in paa_representation])
            sax_words.append(sax_word)


        return sax_words


    def numerosity_reduction(self, sax_words):
        reduced_sax_words = [sax_words[0]]
        for i in range(1, len(sax_words)):
            if sax_words[i] != sax_words[i - 1]:
                reduced_sax_words.append(sax_words[i])
        return reduced_sax_words


    def map_sax_to_subsequences(self, reduced_sax_words, original_time_series):
        subsequences = []
        for idx, word in enumerate(reduced_sax_words):
            start_index = idx  # Adjusted to use idx directly
            if start_index + self.window_size <= len(original_time_series):
                subsequence = original_time_series[start_index:start_index + self.window_size]
                subsequences.append((word, subsequence))
        return subsequences


    def convert_to_terminal(self, element):
        return Terminal(element)


    def run_sequitur(self, data):
        first_rule = Rule(self)  # Pass self (AnomalyDetection instance)
        for element in data:
            terminal = self.convert_to_terminal(element)
            first_rule.insert_after(terminal)
            first_rule.last().check()


        return first_rule.get_rules()


    def count_occurrences(self):
        all_floats = np.concatenate([tup[1] for tup in self.mapped_subsequences])
        for element in all_floats:
            match_index = np.where(self.decision_scores_[:, 0] == element)[0]
            if match_index.size > 0:
                self.decision_scores_[match_index[0], 1] += 1


    def print_result(self):
        np.set_printoptions(precision=8, suppress=True)
        print("-----------------Density Rule Curve results-----------------")
        print(self.decision_scores_)  # Results of counting


    # Distance Calculation: Normalize and calculate the distance
    def z_norm(self, series):
        mean = np.mean(series)
        std_dev = np.std(series)
        if std_dev > 1e-6:  # Avoid division by zero
            return (series - mean) / std_dev
        else:
            return series


    def calculate_distance(self, subsequence1, subsequence2):
        return np.linalg.norm(np.array(subsequence1) - np.array(subsequence2))


    def find_discord(self):
        outer_intervals = range(len(self.mapped_subsequences))
        inner_intervals = range(len(self.mapped_subsequences))
       
        best_so_far_dist = 0
        best_so_far_loc = None
        print("-----------------Rare Rule Anomaly results-----------------")
        print("Outer Intervals:", list(outer_intervals))
        print("-------------------------------------------")
        print("Inner Intervals:", list(inner_intervals))

        for outer_idx in outer_intervals:
            p = self.mapped_subsequences[outer_idx][1]  # Get the subsequence corresponding to outer_idx
            nearest_neighbor_dist = float('inf')

            for inner_idx in inner_intervals:
                if abs(outer_idx - inner_idx) >= len(p):  # Ensure that subsequences do not overlap
                    q = self.mapped_subsequences[inner_idx][1]  # Get the subsequence corresponding to inner_idx

                    current_dist = self.calculate_distance(p, q)


                    if current_dist < nearest_neighbor_dist:
                        nearest_neighbor_dist = current_dist
                        print("Comparing:\n", p, "AND", q)  # Debugging output
                        print("NN DISTANCE", nearest_neighbor_dist)

                    if current_dist < best_so_far_dist:
                        break

            if nearest_neighbor_dist > best_so_far_dist:
                best_so_far_dist = nearest_neighbor_dist
                best_so_far_loc = outer_idx
        print(f"\nDiscord found at index {best_so_far_loc} with a distance of {best_so_far_dist}")

        # Plotting the discord
        plt.figure(figsize=(10, 6))
        plt.plot(self.z_normalized_ts, label="Time Series")
        plt.axvline(x=best_so_far_loc, color='r', linestyle='--', label=f"Discord at {best_so_far_loc}")
        plt.legend()
        plt.title("Discord Detection in Time Series")
        plt.show()

class Rule:
    def __init__(self, anomaly_detector):
        self.number = anomaly_detector.numRules
        self.the_guard = Guard(self)
        self.count = 0
        self.index = 0
        anomaly_detector.numRules += 1  # Increment numRules on initialization

    def first(self):
        return self.the_guard.n

    def last(self):
        return self.the_guard.p

    def insert_after(self, to_insert):
        self.last().insert_after(to_insert)

    def get_rules(self):
        rules = []
        processedRules = 0
        text = ''
        text += "Usage\tRule\n"
        rules.append(self)

        while (processedRules < len(rules)):
            currentRule = rules[processedRules]
            text += " " + str(currentRule.count) + '\tR' + str(processedRules) + ' -> '
            sym = currentRule.first()

            while True:
                if sym.is_guard():
                    break
                if sym.is_nonterminal():
                    rule = sym.r
                    if len(rules) > rule.index and rules[rule.index] == rule:
                        index = rule.index
                    else:
                        index = len(rules)
                        rule.index = index
                        rules.append(rule)
                    text += 'R' + str(index)
                else:
                    if sym.value == ' ':
                        text += '_'
                    elif sym.value == '\n':
                        text += '\\n'
                    else:
                        text += str(sym.value)
                text += ' '
                sym = sym.n
            text += '\n'
            processedRules += 1
        return text

class Symbol:
    def __init__(self):
        self.num_terminals = 100000
        self.prime = 2265539
        self.value = ' '
        self.n = None
        self.p = None

    def clone(self):
        sym = Symbol()
        sym.value = self.value
        sym.n = self.n
        sym.p = self.p
        return sym

    def join(self, left, right):
        if left.n is not None:
            left.delete_digram()
            if (right.p is not None and right.n is not None and
                    right.value == right.p.value and
                    right.value == right.n.value):
                digrams[str(right.value) + str(right.n.value)] = right
            if (left.p is not None and left.n is not None and
                    left.value == left.p.value and
                    left.value == left.n.value):
                digrams[str(left.p.value) + str(left.value)] = left.p
        left.n = right
        right.p = left

    def insert_after(self, to_insert):
        self.join(to_insert, self.n)
        self.join(self, to_insert)


    def digram(self):
        return str(self.value) + str(self.n.value)

    def delete_digram(self):
        if self.n.is_guard(): return


        if self.digram() in digrams:
            dummy = digrams[self.digram()]
            if dummy == self:
                del digrams[self.digram()]


    def is_guard(self):
        return False

    def is_nonterminal(self):
        return False


    def check(self):
        if self.n.is_guard():
            return False


        if self.digram() not in digrams:
            digrams[self.digram()] = self
            return False


        found = digrams[self.digram()]
        if found.n != self:
            self.match(self, found)
        return True


    def cleanup(self):
        pass


    def substitute(self, r):
        self.cleanup()
        self.n.cleanup()
        self.p.insert_after(NonTerminal(r))
        if not self.p.check(): self.p.n.check()



    def match(self, newD, matching):
        if matching.p.is_guard() and matching.n.n.is_guard():
            r = (matching.p).r
            newD.substitute(r)
        else:
            r = Rule(newD)  # Create a new Rule with the current newD
            first = newD.clone()
            second = newD.n.clone()
            r.the_guard.n = first
            first.p = r.the_guard
            first.n = second
            second.p = first
            r.the_guard.cleanup()

class Terminal(Symbol):
    def __init__(self, value):
        super().__init__()
        self.value = value


    def cleanup(self):
        self.join(self.p, self.n)
        self.delete_digram()


    def clone(self):
        sym = Terminal(self.value)
        sym.p = self.p
        sym.n = self.n
        return sym

class NonTerminal(Symbol):
    def __init__(self, r):
        super().__init__()
        self.r = r
        self.r.count += 1
        self.value = self.r.number


    def clone(self):
        sym = NonTerminal(self.r)
        sym.p = self.p
        sym.n = self.n
        return sym

    def cleanup(self):
        self.join(self.p, self.n)
        self.delete_digram()
        self.r.count -= 1


    def is_nonterminal(self):
        return True


    def expand(self):
        self.join(self.p, self.r.first())
        self.join(self.r.last(), self.n)

class Guard(Symbol):
    def __init__(self, r):
        super().__init__()
        self.r = r
        self.value = ' '
        self.p = self
        self.n = self

    def cleanup(self):
        self.join(self.p, self.n)


    def is_guard(self):
        return True


    def check(self):
        return False


if __name__ == '__main__':

    def run_GrammaViz(data=None, window_size=12, alphabet_size=4, word_size=4):
        # Initialize the anomaly detection with specified parameters
        clf = GrammaViz(window_size=window_size,
                        alphabet_size=alphabet_size,
                        word_size=word_size)
        clf.fit(data)
        score = clf.decision_scores_
        return score

    length = 100
    anomaly_positions = [49]
    anomaly_value = 5

    #Generate a custom time series with an anomaly, uncomment the next 3 lines for custom data (optional)
    custom_data = np.sin(np.linspace(0, 4 * np.pi, length)) + 0.1 * np.random.randn(length)
    for position in anomaly_positions:
        custom_data[position] += anomaly_value
    print('custom_data: ', custom_data.shape)

    # Call the function with the custom data
    results = run_GrammaViz(data=custom_data, window_size=10, alphabet_size=5, word_size=3)
    print('results: ', results.shape)
