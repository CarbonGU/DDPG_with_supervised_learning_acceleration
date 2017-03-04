from SL_gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json
import copy
from ReplayBuffer import ReplayBuffer
from SL_ActorNet import ActorNetwork
from SL_CriticNet import CriticNetwork
import snakeoil3_gym as snakeoil3
from OU import OU
from FULL_CRUISE_SPEED import CRUISE_SPEED
CRUISE_SPEED = CRUISE_SPEED()
from supervisor import SUPERVISOR

PI= 3.14159265359
OU = OU()       #Ornstein-Uhlenbeck Process
SUPERVISOR = SUPERVISOR()

def simuSupervisor(states):
    angle    = np.asarray([[e[1]] for e in states])
    trackPos = np.asarray([[e[2]] for e in states])

    angle    = angle * 3.1416
    trackPos = trackPos * 1.0

    steer    = angle * 10 / PI
    steer   -= trackPos * 0.10

    return steer


def playGame(train_indicator=1):    #1 means Train, 0 means simply Run

    BUFFER_SIZE     = 100000
    BATCH_SIZE      = 32
    GAMMA           = 0.99

    TAU             = 0.001     #Target Network HyperParameters
    LRA             = 0.0001    #
    # Learning rate for Actor
    LRC             = 0.001     #Lerning rate for Critic
    #PI             = 3.14

    action_dim      = 1  #Kp/Ki
    state_dim       = 3  #of sensors input
    np.random.seed(1337)

    EXPLORE         = 50000.
    episode_count   = 500
    max_steps       = 500

    sl_param_k      = 0.0

    reward          = 0
    done            = False
    step            = 0
    epsilon         = 1
    step0           = 0
    #Tensorflow GPU optimization
    config                          = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess                            = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor     = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic    = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff      = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    env       = TorcsEnv(vision=False, throttle=False, gear_change=False)

    #Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("SL_actormodel.h5")
        critic.model.load_weights("SL_criticmodel.h5")
        actor.target_model.load_weights("SL_actormodel.h5")
        critic.target_model.load_weights("SL_criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("TORCS Experiment Start.")

    file1 = open('SL_reward_output.txt','a')
    file1.write("****new record****\n")
    file1.close()

    file2 = open('SL_steps_output.txt','a')
    file2.write("****new record****\n")
    file2.close()

    for i in range(episode_count):
        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)
        else:
            ob = env.reset()

        s_t              = np.hstack((ob.speedX, ob.angle, ob.trackPos)) #  5 dims
        total_reward     = 0.
        for j in range(max_steps):
            loss         = 0
            epsilon     -= 1.0 / EXPLORE
            a_t          = np.zeros([1,action_dim])
            noise_t      = np.zeros([1,action_dim])
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            #normalization

            noise_t[0][0]  = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            a_t[0][0]      = a_t_original[0][0] + noise_t[0][0]

            angle_error    = ob.angle
            trackPos_error = ob.trackPos

            ob, r_t, done, info = env.step(a_t[0], angle_error, trackPos_error)
            print("angle-error:", angle_error, "trackPos-error:", trackPos_error)
            s_t1                = np.hstack((ob.speedX, ob.angle, ob.trackPos))
            buff.add(s_t, a_t[0], r_t, s_t1, done)

            #Do the batch update
            batch      = buff.getBatch(BATCH_SIZE)
            states     = np.asarray([e[0] for e in batch])
            actions    = np.asarray([e[1] for e in batch])
            rewards    = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones      = np.asarray([e[4] for e in batch])
            y_t        = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])
            sl_actions      = simuSupervisor(states)
            action_error    = sl_actions - actions
            #print("sl_action:", sl_actions)
            #print("actions:", actions)
            #print("action_error:", action_error)

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]

            if step < EXPLORE / 4.0:
                sl_param_k += 0.8 / (EXPLORE / 4.0)
            else:
                sl_param_k  = 0.8

            print("sl_param_k:", sl_param_k)

            if (train_indicator):
                loss        += critic.model.train_on_batch([states,actions], y_t)
                a_for_grad   = actor.model.predict(states)
                grads        = critic.gradients(states, a_for_grad)
                grads        = np.dot(grads, sl_param_k)
                action_error = np.dot(action_error, 1 - sl_param_k)
                grads       += action_error

                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t           = s_t1
            print(" ")
            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
            print(" ")

            step += 1
            if done:
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("SL_actormodel.h5", overwrite=True)
                with open("SL_actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("SL_criticmodel.h5", overwrite=True)
                with open("SL_criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        file1 = open('SL_reward_output.txt','a')
        print>>file1,'%0.4f' %total_reward
        file1.close()
        delta_step = step -step0
        file2 = open('SL_steps_output.txt', 'a')
        print>>file2,'%d' %delta_step
        file2.close()
        step0 = step

        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame()
