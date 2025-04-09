import random
import copy
from SnakeGameAI import SnakeGameAI

POPULATION_SIZE = 50 #50
GENE_LENGTH = 100
MUTATION_RATE = 0.05
NUM_GENERATIONS = 20 #200

#genel olarak yılanın hareketlerini random deneyip en iyi performans gösteren bireyleri seçer.
#bu bireyleri çaprazlar ve mutasyona uğratır
class GeneticAgent:
    def __init__(self):
        self.population = [self.random_individual() for _ in range(POPULATION_SIZE)]

    def random_individual(self):
        return [random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) for _ in range(GENE_LENGTH)]#random hareket seçiyoruz

    def evaluate_fitness(self, individual):
        game = SnakeGameAI()
        total_reward = 0
        last_distance = self.get_distance(game.head, game.food)

        for action in individual:
            reward, game_over, score = game.play_step(action)
            new_distance = self.get_distance(game.head, game.food)

            if new_distance < last_distance:
                total_reward += 1  # Yaklaştıysa ödül
            elif new_distance > last_distance:
                total_reward -= 1  # Uzaklaştıysa ceza

            last_distance = new_distance

            if game_over:
                break

            total_reward += reward  # Yem yediyse +10 gibi eklenir

        fitness = game.score * 100 + total_reward# yılanın skoru ve yeme yaklaşma uzalaşma durumuna göre fitness hesaplanır
        return fitness, game.score

    def get_distance(self, head, food):
        return abs(head[0] - food[0]) + abs(head[1] - food[1])  # Manhattan mesafesi


    #en yüksek fitness skoruna sahip bireyler parent olarak seçilir
    def select_parents(self, fitness_scores):
        sorted_population = sorted(zip(fitness_scores, self.population), key=lambda x: x[0], reverse=True)
        parents = [ind for _, ind in sorted_population[:10]]  # En iyi 10 bireyi seç
        return parents

    def crossover(self, parent1, parent2):
        idx = random.randint(1, GENE_LENGTH - 2)
        child = parent1[:idx] + parent2[idx:] # 2 tane parentin genleri birleşiyor
        return child

    def mutate(self, individual):
        for i in range(GENE_LENGTH):
            if random.random() < MUTATION_RATE:
                individual[i] = random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])#random mutasyona uğratıyoruz
        return individual

    #her nesilde bireyler değerlendirilir, seçilir, çaprazlanır ve mutasyona uğratılır.
    def evolve(self):
        for gen in range(NUM_GENERATIONS):# bu kadar nesil boyunca döngü çalışıyor

            #burada bireyin oyunda ne kadar başarılı olunduğu ölçülüp bi fitness skoru dödürülüyor
            fitness_scores = []#bireylerin skorları burada
            for ind in self.population:
                fitness, score = self.evaluate_fitness(ind)
                fitness_scores.append(fitness)
            best_fitness = max(fitness_scores)
            best_score = max([self.evaluate_fitness(ind)[1] for ind in self.population])
            print(f"Gen {gen+1}/{NUM_GENERATIONS} | En iyi fitness: {best_fitness:.2f} | Skor: {best_score}", flush=True)

            #fitness skoruna göre en iyi bireyler seçiliyor
            parents = self.select_parents(fitness_scores)
            new_population = [] # yeni nesil oluşturuluyor

            #yeni nesil oluşturulurken çaprazlama ve mutasyon işlemleri yapılıyor
            while len(new_population) < POPULATION_SIZE:
                p1, p2 = random.sample(parents, 2)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)

            self.population = new_population

    #en iyi bireyler seçilip test ediliyor
    def test_best(self):
        best_ind = max(self.population, key=lambda ind: self.evaluate_fitness(ind)[0])
        game = SnakeGameAI()
        while True:
            for action in best_ind:
                reward, game_over, score = game.play_step(action)
                if game_over:
                    print("Test bitti! Skor:", score)
                    return
                
if __name__ == '__main__':
    agent = GeneticAgent()
    agent.evolve()
    agent.test_best()