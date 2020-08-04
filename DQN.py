import gym
import numpy as np
import tensorflow as tf

class ReplayBuffer:
    """Um buffer que armazena transições e permite a amostragem de transições aleatórias."""

    def __init__(self, max_size, state_dim):
        """Cria um replay buffer.
        Args:
            max_size (int): número máximo de transições armazenadas pelo buffer.
        """
        self.max_size = max_size

        self.mem_cntr = 0

        self.state_memory = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.state2_memory = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.action_memory = np.zeros(self.max_size, dtype = np.int32)
        self.terminal_memory = np.zeros(max_size, dtype=np.float32)

    def add_transition(self, transition):
        """Adiciona uma transição ao replay buffer.
        Args:
            transition (tuple): uma tupla representando a transição.
        """
        index = self.mem_cntr % self.max_size
        self.state_memory[index] = transition[0]
        self.state2_memory[index] = transition[1]
        self.reward_memory[index] = transition[2]
        self.action_memory[index] = transition[3]
        self.terminal_memory[index] = transition[4]

        self.mem_cntr += 1

    def sample_transitions(self, num_samples):
        """Retorna uma lista com transições amostradas aleatoriamente.
        Args:
            num_samples (int): o número de transições desejadas.
        """
        max_size = min(self.mem_cntr, self.max_size)
        sample = np.random.choice(max_size, num_samples, replace=False)

        state = self.state_memory[sample]
        state2 = self.state2_memory[sample]
        reward = self.reward_memory[sample]
        action = self.action_memory[sample]
        done = self.terminal_memory[sample]

        transition = (state, state2, reward, action, done)

        return transition

    def get_size(self):
        """Retorna o número de transições armazenadas pelo buffer."""
        return self.mem_cntr

    def get_max_size(self):
        """Retorna o número máximo de transições armazenadas pelo buffer."""
        return self.max_size


class DQNAgent:
    """Implementa um agente de RL usando Deep Q-Learning."""

    def __init__(self, state_dim, action_dim, architecture,
                 buffer_size=100_000,
                 batch_size=128,
                 gamma=1.00):
        """Cria um agente de DQN com os hiperparâmetros especificados
        Args:
            state_dim (int): número de variáveis de estado.
            action_dim (int): número de ações possíveis.
            architecture (list of float, optional): lista com o número de neurônios
                                                    de cada camada da DQN.
            buffer_size (int, optional): tamanho máximo do replay buffer.
            batch_size (int, optional): número de transições utilizadas por batch.
            gamma (float, optional): fator de desconto utilizado no calculo do retorno.
        """
        self.memory = ReplayBuffer(buffer_size, state_dim)

        layers = tf.keras.layers
        DQN = tf.keras.models.Sequential()
        DQN.add(layers.Input(shape=(state_dim)))
        for n in architecture:
            DQN.add(layers.Dense(n, activation='relu'))
        DQN.add(layers.Dense(action_dim, activation='linear'))
        DQN.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')
        self.model = DQN

        self.actions = [i for i in range(action_dim)]
        self.gamma = gamma
        self.batch_size = batch_size

    def act(self, state, epsilon=0):
        """Retorna a ação tomada pelo agente no estado `state`.
        Args:
            state: o estado do ambiente.
            epsilon (int, optional): o valor de epsilon a ser considerado
                                     na política epsilon-greedy.
        """
        if np.random.random_sample() < epsilon:
            action = np.random.choice(self.actions)
        else:
            s = np.array([state])
            actions = self.model.predict(s)
            action = np.argmax(actions)
        return action
        
    def save_memory(self, s, s2, r, a, done):
        self.memory.add_transition((s, s2, r, a, done))

    def optimize(self):
        """Roda um passo de otimização da DQN.
        Obtém `self.batch_size` transições do replay buffer
        e treina a rede neural. Se o replay buffer não tiver
        transições suficientes, a função não deve fazer nada.
        """
        if self.memory.get_size() < self.batch_size:
            return
        s, s2, r, a, done = self.memory.sample_transitions(self.batch_size)
        q_eval = self.model.predict(s)
        q_next = self.model.predict(s2)
        q_target = np.copy(q_eval)

        q_target[:,a]= r + self.gamma * np.max(q_next, axis=1) * (1 - done)
        self.model.train_on_batch(s, q_target)

if __name__ == '__main__':
    # Crie o ambiente 'pong:turing-easy-v0'
    env = gym.make('pong:turing-easy-v0').env

    # Hiperparâmetros da política epsilon-greedy
    initial_eps = 1
    min_eps = 0.01
    eps_decay = .85
    eps = initial_eps
    gamma = .995

    # Número total de episódios
    num_episodes = 25

    scores = []

    agent = DQNAgent(action_dim=3,
                     state_dim=4,
                     architecture=[32, 32],
                     batch_size=512,
                     gamma=gamma)

    for episode in range(num_episodes):
        # Rodar o episódio, da mesma forma que com o agente aleatório.
        done = False
        score = 0
        t = 0
        obs = env.reset()

        while not done:
            action = agent.act(obs, eps)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.save_memory(obs, obs_, reward, action, done)
            obs = obs_
            if t%32 == 0:  agent.optimize()
            t += 1
        if eps > min_eps: eps *= eps_decay
        else: eps = min_eps
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print("jogos: ", episode, "pontuação: %.2f" % score, "pontuação média: %.2f" % avg_score, "epsilon: %.2f" % eps)

    # Fechando o ambiente
    env.close()