from collections import defaultdict


class Classifier(object):
    def __init__(self):
        self.features = defaultdict(int)
        self.labels = defaultdict(int)
        self.feature_counts = defaultdict(lambda: defaultdict(int))
        self.total_count = 0

    def train(self, features, labels):
        """This method updates the counts of features associated with
        the specified lables.

        Args:
            features ([string]). A list of strings associated with the given
                labels. Typically a tokenized string.
            labels: The labels associated with the features.

        Example:
            >>> c = Classifier()
            >>> c.train(["please", "send", "money", "dear", "sir"], "spam")
        """
        for label in labels:
            for feature in features:
                self.feature_counts[feature][label] += 1
                self.features[feature] += 1

            self.labels[label] += 1
        self.total_count += 1

    def feature_probability(self, feature, label):
        """Calculates the probability of a feature occurring in a given
        label.

        Args:
            feature: The word being searched for, e.g., "money".
            label: The messages being searched through, "spam" or "ham".

        Returns:
            A probability that our feature occurs in messages with the
            chosen label

        Example:
            >>> c = Classifier()
            >>> c.feature_probability("money", "spam")
            0.12
        """
        feature_count = self.feature_counts[feature][label]
        label_count = self.labels[label]
        if feature_count and label_count:
            return float(feature_count) / label_count
        return 0

    def weighted_probability(self, feature, label, weight=1.0, ap=0.5):
        """This method computes a weighted probability that a feature is
        associated with a given label.

        Args:
            feature (string)
            label (string)
            weight (float): A weight constant.
            ap (float): A constant representing average probability. 0.5 is the
                default, representing the probability of a feature being associated
                with "spam" or "ham" if both labels are equally probable.

        Returns:
            A weighted probability that a feature matches a label.
        """
        p_initial = self.feature_probability(feature, label)
        feature_total = self.features[feature]
        return float((weight * ap) + (feature_total * p_initial)) / (weight + feature_total)

    def document_probability(self, features, label):
        """Measures the probability that a set of features matches a label
        by multiplying individual feature probabilities together.

        Args:
            features ([string])
            label (string)

        Returns:
            Weighted probability representing the probability that a document has
            a certain label ("spam" or "ham")
        """
        p = 1
        for feature in features:
            p *= self.weighted_probability(feature, label)
        return p

    def probability(self, features, label):
        if not self.total_count:
            return 0
        label_prob = float(self.labels[label]) / self.total_count
        doc_prob = self.document_probability(features, label)
        return doc_prob * label_prob

    def classify(self, features, limit=5):
        """TODO
        """
        probability = {}
        for label in self.labels.keys():
            probs[label] = self.probability(features, label)
        return sorted(probs.items(), key=lambda (k,v): v, reverse=True)[:limit]
