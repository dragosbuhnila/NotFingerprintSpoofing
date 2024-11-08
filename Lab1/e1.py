import sys

class Athlete_Score:
    final_score = 0
    def __init__(self, name, country, scores):
        self.name = name
        self.country = country
        self.scores = scores

    def __str__(self):
        return f"{self.name} {self.country} {self.scores} {self.final_score}"

    def __lt__(self, other):
        return self.final_score < other.final_score

    def __eq__(self, other):
        return self.final_score == other.final_score


def main():
    # (1) Reading a file using the *with* block
    with open(sys.argv[1], 'r') as scorefile:
        athlete_scores = []

        for line in scorefile:
            line = line.strip().split(" ")

            full_name = f"{line[0]} {line[1]}"
            country = line[2]
            scores = line[3:]

            athlete_score = Athlete_Score(full_name, country, scores)
            athlete_scores.append(athlete_score)

    for score in athlete_scores:
        score.scores.sort()
        score.scores.pop()
        score.scores.pop(0)

        total = 0.0
        for singlescore in score.scores:
            total += float(singlescore)

        score.final_score = total

    athlete_scores.sort(reverse=True)

    top_three = athlete_scores[:3]
    i = 1
    print()
    print("Final Ranking:")
    for score in top_three:
        print(f"{i}: {score.name} - Score: {score.final_score}")
        i += 1

    country_points = {}
    for score in athlete_scores:
        if score.country in country_points:
            country_points[score.country] += score.final_score
        else:
            country_points[score.country] = score.final_score
    country_points = sorted(country_points.items(), key=lambda x: x[1], reverse=True)

    top_country = country_points[0]
    print()
    print("Best Country:")
    print(f"{top_country[0]} - Total Score: {top_country[1]}")







if __name__ == "__main__":
    main()