import torch
import random
import numpy as np
from collections import deque
from SnakeGameAI import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 #learning rate

# yılanın çevreyle etkileşimini, öğrenme sürecini ve eğitim dögüsünü yönetiyor
class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0.9 #discount rate(1'den küçük olacak, buna göre değiştirebilirim)
        self.memory = deque(maxlen=MAX_MEMORY)#popleft()
        self.model = Linear_QNet(11, 256, 3)#size of the state, hidden size, output
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    #yılanın anlık durumunu hesaplıyor
    def get_state(self, game):
        head = game.snake[0]

        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #önde tehlike varsa
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            #sağda tehlike varsa
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            #solda tehlike varsa
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            
            #hareket yönlerimiz
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #yemek yerleri
            game.food.x < game.head.x, #yemek solda
            game.food.x > game.head.x, #yemek sağda
            game.food.y < game.head.y, #yemek yukarıda
            game.food.y > game.head.y #yemek aşağıda
        ]

        return np.array(state, dtype=int) #booleanları 0 ve 1'e çevirmek için

    def remember(self, state, action, reward, next_sate, done):
        self.memory.append((state, action, reward, next_sate, done))#deneyimleri (state, action, reward, next_sate, done) hafızaya kaydetti

    #hafızadan rastgele bir batch seçerek modeli eğitiyor
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #tuple listesi
        else:
            mini_sample = self.memory

        states, actions, rewards, next_sates, dones = zip(*mini_sample) #buna bak
        self.trainer.train_step(states, actions, rewards, next_sates, dones)

    # modeli tek bir adımda eğitiyor   
    def tarin_short_memory(self, state, action, reward, next_sate, done):
        self.trainer.train_step(state, action, reward, next_sate, done)

    ## epsilon-greedy algoritması kullanarak bir eylem seçiyor
    def get_action(self, state):
        #tradeoff exploration and exploitation

        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon: #rastgele hareket
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()#modelin önerdiği hareket
            final_move[move] = 1

        return final_move

#sonsuz bir döngüde yılanı eğitiyor
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.tarin_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game ', agent.n_games, 'Score ', score, 'Record: ', record)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()

# bu şekilde de yapabilirmişiz zip yerine
'''for state, action, reward, next_sate, done in mini_sample:
           self.trainer.train_step(state, action, reward, next_sate, done)'''
