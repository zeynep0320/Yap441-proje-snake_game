import numpy as np
import random
from SnakeGameAI import SnakeGameAI, Direction, Point, BLOCK_SIZE
import matplotlib.pyplot as plt

class GeneticAgent:
    def __init__(self, population_size=50, mutation_rate=0.2, max_moves=500):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_moves = max_moves
        # 11 state özelliği × 3 action için gen boyutu
        self.genome_size = 11 * 3
        self.population = [self._random_genome() for _ in range(population_size)]
        self.best_fitness_history = []

    def _random_genome(self):
        """Rastgele gen oluşturma"""
        return np.random.uniform(-1, 1, size=(self.genome_size,))

    def _get_state(self, game):
        """Gelişmiş durum temsili"""
        head = game.head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Önümde engel var mı?
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Sağımda engel var mı?
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Solumda engel var mı?
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Hareket yönü
            dir_l, dir_r, dir_u, dir_d,

            # Yem konumu
            game.food.x < game.head.x,  # yem sola mı?
            game.food.x > game.head.x,  # yem sağa mı?
            game.food.y < game.head.y,  # yem yukarı mı?
            game.food.y > game.head.y   # yem aşağı mı?
        ]
        return np.array(state, dtype=int)

    def _choose_action(self, state, genome):
        """Genoma göre action seçimi"""
        weights = genome.reshape(3, 11)  # 3 action × 11 state özelliği
        
        # Action değerlerini hesapla
        action_values = np.dot(weights, state)
        
        # Softmax ile olasılık dağılımı
        exp_values = np.exp(action_values - np.max(action_values))
        softmax_probs = exp_values / np.sum(exp_values)
        
        # Olasılıklara göre action seç
        action_index = np.random.choice([0, 1, 2], p=softmax_probs)
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]][action_index]

    def _evaluate_fitness(self, genome):
        """Gelişmiş fitness hesaplama"""
        game = SnakeGameAI()
        total_reward = 0
        state = self._get_state(game)
        prev_distance = abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)
        steps_without_food = 0
        
        for _ in range(self.max_moves):
            action = self._choose_action(state, genome)
            reward, game_over, _ = game.play_step(action)
            
            # Yeme olan mesafe
            new_distance = abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)
            
            # Mesafe ödül/ceza sistemi
            if new_distance < prev_distance:
                reward += 2  # Yeme yaklaşıyorsa ödül
            else:
                reward -= 1  # Uzaklaşıyorsa ceza
                
            total_reward += reward
            prev_distance = new_distance
            
            if game_over:
                break
                
            state = self._get_state(game)
            
        # Skora göre büyük ödül
        total_reward += game.score * 50
        
        # Uzun süre yem yemeden dolaşmaya ceza
        if game.score == 0:
            total_reward -= 10
            
        return total_reward

    def _select_best(self, num_parents=10):
        """En iyi bireyleri seçme (elitizm)"""
        fitness_scores = [(genome, self._evaluate_fitness(genome)) for genome in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        return [genome for genome, _ in fitness_scores[:num_parents]]

    def _crossover(self, parent1, parent2):
        """İki noktalı çaprazlama"""
        size = len(parent1)
        pt1, pt2 = sorted(random.sample(range(size), 2))
        child = np.concatenate((parent1[:pt1], parent2[pt1:pt2], parent1[pt2:]))
        return child

    def _mutate(self, genome):
        """Rastgele mutasyon"""
        for i in range(len(genome)):
            if random.random() < self.mutation_rate:
                genome[i] += np.random.uniform(-0.5, 0.5)
                genome[i] = np.clip(genome[i], -2, 2)  # Değerleri sınırla
        return genome

    def evolve(self, generations=200, target_fitness=500, show_plot=True):
        """Genetik algoritma ile evrim"""
        self.best_fitness_history = []
        no_improvement_for = 0
        best_fitness = float('-inf')

        for gen in range(generations):
            # En iyi bireyleri seç (elitizm)
            best_individuals = self._select_best(num_parents=self.population_size // 2)
            new_population = best_individuals.copy()

            # Yeni bireyler oluştur
            while len(new_population) < self.population_size:
                p1, p2 = random.sample(best_individuals, 2)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_population.append(child)

            self.population = new_population

            # İlerlemeyi kaydet
            current_best_fitness = max(self._evaluate_fitness(g) for g in self.population)
            self.best_fitness_history.append(current_best_fitness)
            print(f"Nesil {gen + 1}/{generations} - En iyi fitness: {current_best_fitness}")

            # Erken durdurma kontrolü
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                no_improvement_for = 0  # İyileşme varsa sıfırla
            else:
                no_improvement_for += 1  # İyileşme yoksa artır

            if best_fitness >= target_fitness or no_improvement_for > 10:
                print(f"Erken durdurma: {gen + 1}. nesilde durduruldu.")
                break

        # Fitness gelişim grafiği
        if show_plot:
            plt.plot(self.best_fitness_history)
            plt.title("Fitness Gelişimi")
            plt.xlabel("Nesil")
            plt.ylabel("Fitness Değeri")
            plt.show()

        print("Evrim tamamlandı!")
        return self.population[0]  # En iyi bireyi döndür
    
    
if __name__ == "__main__":
    # Parametrelerle denemeler yapabilirsiniz
    agent = GeneticAgent(
        population_size=50,
        mutation_rate=0.15,
        max_moves=1000
    )
    
    # 200 nesil çalıştır
    best_genome = agent.evolve(generations=200)
    
    # En iyi bireyi test et
    print("En iyi birey testi:")
    game = SnakeGameAI()
    while True:
        state = agent._get_state(game)
        action = agent._choose_action(state, best_genome)
        _, game_over, score = game.play_step(action)
        
        if game_over:
            print(f"Oyun bitti! Skor: {score}")
            break


'''import numpy as np
import random
import time
from SnakeGameAI import SnakeGameAI, Direction, Point, BLOCK_SIZE

class GeneticAgent:
    def __init__(self, population_size=30, mutation_rate=0.1, max_moves=200):
        # Temel parametreler
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_moves = max_moves
        
        # Genom boyutu (11 giriş × 3 çıkış)
        self.genome_size = 11 * 3
        self.population = [self._random_genome() for _ in range(population_size)]
        self.best_fitness_history = []

    def _random_genome(self):
        """-1 ile 1 arasında rastgele genler"""
        return np.random.uniform(-1, 1, size=(self.genome_size,))

    def _get_state(self, game):
        """Basitleştirilmiş durum temsili"""
        head = game.head
        return np.array([
            # Engel bilgileri (ön, sağ, sol)
            game.is_collision(Point(head.x + BLOCK_SIZE, head.y)),  # Önümde engel
            game.is_collision(Point(head.x, head.y - BLOCK_SIZE)),  # Sağımda engel
            game.is_collision(Point(head.x, head.y + BLOCK_SIZE)),  # Solumda engel
            
            # Hareket yönü
            game.direction == Direction.LEFT,
            game.direction == Direction.RIGHT,
            game.direction == Direction.UP,
            game.direction == Direction.DOWN,
            
            # Yem konumu
            game.food.x < game.head.x,  # Yem solda
            game.food.x > game.head.x,  # Yem sağda
            game.food.y < game.head.y,  # Yem yukarıda
            game.food.y > game.head.y   # Yem aşağıda
        ], dtype=int)

    def _choose_action(self, state, genome):
        """Genoma göre hareket seçimi"""
        weights = genome.reshape(3, 11)  # 3 hareket × 11 giriş
        output = np.dot(weights, state)
        return [
            [1, 0, 0],  # Düz git
            [0, 1, 0],  # Sağa dön
            [0, 0, 1]   # Sola dön
        ][np.argmax(output)]  # En yüksek değerli hareketi seç

    def _evaluate_fitness(self, genome):
        """Basit ve etkili fitness fonksiyonu"""
        game = SnakeGameAI()
        state = self._get_state(game)
        score = 0
        
        for _ in range(self.max_moves):
            action = self._choose_action(state, genome)
            reward, game_over, current_score = game.play_step(action)
            
            if game_over:
                return current_score * 10  # Skor × 10
            
            score = current_score
            state = self._get_state(game)
        
        return score * 10

    def _select_parents(self):
        """En iyi %50'yi seç"""
        fitnesses = [self._evaluate_fitness(g) for g in self.population]
        sorted_pop = sorted(zip(self.population, fitnesses), key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_pop[:self.population_size//2]]

    def _crossover(self, parent1, parent2):
        """Tek noktalı çaprazlama"""
        point = random.randint(0, len(parent1)-1)
        return np.concatenate((parent1[:point], parent2[point:]))

    def _mutate(self, genome):
        """Kontrollü mutasyon"""
        for i in range(len(genome)):
            if random.random() < self.mutation_rate:
                genome[i] += random.uniform(-0.5, 0.5)
                genome[i] = np.clip(genome[i], -1, 1)
        return genome

    def evolve(self, generations=50):
        """Ana eğitim döngüsü"""
        print("Eğitim başlıyor...")
        
        for gen in range(generations):
            start_time = time.time()
            
            # 1. Ebeveyn seçimi
            parents = self._select_parents()
            best_fitness = self._evaluate_fitness(parents[0])
            self.best_fitness_history.append(best_fitness)
            
            # 2. Yeni nesil oluşturma
            new_population = parents.copy()
            while len(new_population) < self.population_size:
                p1, p2 = random.sample(parents, 2)
                child = self._mutate(self._crossover(p1, p2))
                new_population.append(child)
            
            self.population = new_population
            
            # 3. İlerlemeyi göster
            print(f"Nesil {gen+1}/{generations} | En iyi fitness: {best_fitness} | Süre: {time.time()-start_time:.1f}s")
        
        # En iyi bireyi döndür
        best_genome = max(self.population, key=lambda g: self._evaluate_fitness(g))
        print("Eğitim tamamlandı!")
        return best_genome

if __name__ == "__main__":
    # Hızlı test için küçük parametreler
    agent = GeneticAgent(
        population_size=20,
        mutation_rate=0.15,
        max_moves=100
    )
    
    best_genome = agent.evolve(generations=30)
    
    # En iyi bireyi test et
    print("\nTest oyunu başlıyor...")
    game = SnakeGameAI()
    while True:
        state = agent._get_state(game)
        action = agent._choose_action(state, best_genome)
        _, game_over, score = game.play_step(action)
        
        if game_over:
            print(f"Test skoru: {score}")
            break


'''