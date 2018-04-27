
# coding: utf-8

#   #                                               OPEN AI GYM CARTPOLE V1
# 
#    

#  In the following program we shall train a neural network to learn to play the game 'Cart-Pole'.This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson.
# 

# In[1]:


#Lets start by importing all our dependecies
import gym          #OpenAi Game environment
import random
import numpy as np
import tflearn      #Machine learning package used to build out neural network
from tflearn.layers.core import input_data,dropout,fully_connected #Layers needed for the network
from tflearn.layers.estimator import regression     
from statistics import mean , median    #For statistical analysis


# In[2]:


#Defining network parameters

LR=1e-3             #LEARNING RATE
goal_setps= 500     #update to 200 later
score_req = 50      #can update later
initial_games =5000 #No of games to play to generate training data


# In[3]:


#Initializing the game environment and type

env=gym.make('CartPole-v1')  #Importing game env
env.reset()                  #Starts/Re-starts the game Environment 


# ### Getting a feel for Gym

# Now that we have defined out network parameters and initialized our environment,lets see what it looks like.
# The following function makes random moves as it renders the game for 10 episodes.

# In[4]:


def random_games():

    for episode in range(10):
        env.reset()
        score=0

        for t in range(goal_setps):

            env.render()

            action=env.action_space.sample()

            observation , reward , done , info =env.step(action)
            #print(observation , reward , done)
            score+=reward
            if done:
                break


# In[5]:


#Executing this cell will run the above function rendering the game 

random_games()


# ### Generating training data

# To train any ML model we need training data.
# In this case we shall generate our own training data but recording observation for the games we shall play.The following function ' initial_population ' will help us populate the training_data list.We add only the observations of the gamw where we have achieved a score of more than the requried score.

# In[6]:


#This function returns a list of populated training data
def initial_population():
    
    
    
    training_data=[]
    scores=[]
    accepted_scores=[] #Going to accept scores above 50
    
    for _ in range(initial_games):
        #Each new game starts with a fresh score-board and game_memory
        score=0
        game_memory=[]
        prev_observation=[]

        for _ in range(goal_setps):
            
            #In this loop we iterate over fames in each game(episode)
            
            #We pick a random action between 0 and 1 which corresponds to left and right
            action=random.randrange(0,2)
            '''
            This action is used to take a "step" in the env,which returns
            observation(dtype:list),
            reward(dtype:int) > it is 1 if the cart is able to survive the current frame or else its 0
            done(dtype:bool)
            info contains information about the step,if needed can be logged but we arent going to be needing it.
             '''
            observation , reward , done , info =env.step(action)
            
            
            #We shall now store the observation into the game memory along with the action which was taken to obtain that state
            if len(prev_observation) > 0:
                game_memory.append([prev_observation,action])
            #Updating observation and rewarding the scoreaccordingly
            prev_observation = observation
            score += reward
            
            #done = True ,when episode is complete and hence we break from this loop
            if done:
                
                break
        
        #Now we shall only store the data of games where we have a score of more than the requried score.
        
        if score >= score_req:
            accepted_scores.append(score)
            for data in game_memory:
                '''
                Converting to one hot output.
                Where the category to which the data belongs to will be True(i.e- 1) and other categories shall be false(i.e 0)
                We are prefereably using this as we may encounter 
                categorial data of more than 2 categories in future cases.
                '''
                if data[1]==1:
                    output = [0,1]
                elif data[1]==0:
                    output=[1,0]
                #Appending the training data
                training_data.append([data[0],output])
        #Restting the env for next game
        env.reset()
        scores.append(score)
    
    #Statistical data
    print('Avg accepted score',mean(accepted_scores))
    print('Median accepted score',median(accepted_scores))
    


    return training_data
    
     


# ## Building the Neural Network

# We shall now create our neural network model using 
# all our imported layers from tflearn.The following function shall satisfy requriments and returning the requried model.
#  - Our networks shall have 5 hidden layers of sizes 128 , 256 , 512 , 256 , 128  nodes respectively.
#  - We shall we using 'Relu' as out activation function and a dropout of 30% on the nodes.
#  - The output layer shall consist of 2 nodes activated by softmax function.
#  - We shall then perfrom regression on the followinf model to optimize for categorical_crossentropy loss using 'Adam' optimizer.
#  

# In[7]:


def neural_network_model(input_size):
    network = input_data(shape = [None, input_size,1] ,name='input')

    network=fully_connected(network,128,activation='relu')
    network=dropout(network,0.7)

    network=fully_connected(network,256,activation='relu')
    network=dropout(network,0.7)

    network=fully_connected(network,512,activation='relu')
    network=dropout(network,0.7)

    network=fully_connected(network,256,activation='relu')
    network=dropout(network,0.7)

    network=fully_connected(network,128,activation='relu')
    network=dropout(network,0.7)


    network = fully_connected(network , 2 ,activation='softmax')
    network = regression(network,optimizer='adam',learning_rate=LR,loss='categorical_crossentropy',name='targets')

    model=tflearn.DNN(network)

    return model


# ## Time To Train

# Now that we have our training data and our neural network model ready we can finally train it!!
# 

# In[8]:


def train_model(training_data,model=False):
    #model=False is default as we do not expect to have a model on the 1st run,however a pre-trainined model can also be used.
    
    #0th index of training_data contains direction of motion L/R
    X=np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    
    #1st element of training_data contains the output
    y=np.array([i[1] for i in training_data])

    #If we do not have a previously defined model we shall define it here.
    if not model:
        model= neural_network_model(input_size=len(X[0]))

    #The NN is fed the training the data in X and output labels in y to fins a function to map the 2
    model.fit({'input':X},{'targets':y},n_epoch=3,snapshot_step=500,show_metric=True,run_id='openaiCartPoleV1')

    return model


# In[9]:



#Call the above function to train the model

training_data = initial_population()
model = train_model(training_data=training_data)

#In case you wanna save the model for future uses.
#model.save("CartPoleV1.model")


# ## Finally Test Time

# Now that we have built and trained our model lets see how it performs.We will be logging the choices made(L/R) and the scores for validating out model.
# 
# According to the documentation on achieving a score of 195 over 100 games,this problem is considered solved.
# 
# Let's see if we can get there...

# In[10]:


scores=[]
choices=[]

for each_game in range(100):
    score=0
    game_memory=[]
    prev_obs=[]
    env.reset()
    for _ in range(goal_setps):

        #env.render()
        if len(prev_obs) == 0:
            action=random.randrange(0,2)
        else:
            model_prediction=model.predict(prev_obs.reshape(-1,len(prev_obs),1))
            action = np.argmax(model_prediction[0])

        choices.append(action)

        new_obs , reward,done,info=env.step(action)
        prev_obs=new_obs
        game_memory.append([new_obs,action])

        score+=reward



        if done:
            break

    scores.append(score)


# In[13]:



print("Score Sheet \n",scores)
print('Avgerage Score:',float(sum(scores))/len(scores))
print("Choice 1: {} , Choice 2 : {}".format(float(choices.count(1))/len(choices),float(choices.count(0))/len(choices)))


# # Thank you
