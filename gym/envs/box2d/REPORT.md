## Environment
- There are two environment versions: discrete or continuous.
- The landing pad is always at coordinates (0,0). 
- The coordinates are the first two numbers in the state vector.
- Landing outside of the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt.

## Action Space

There are four discrete actions available: 
    1. do nothing 0
    2. fire left orientation engine 1 
    3. fire main engine 2 
    4. fire right orientation engine. 3

For continuous, Action is two floats [main engine, left-right engines].
    1. Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
    2. Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off

## State Space
Total 8 dimensions
1. coordinates of the lander in `x` & `y`
2. its linear velocities in `x` & `y`
3. angle
4. angular velocity
5. two booleans that represent whether each leg is in contact with the ground or not

The lander starts at the top center of the viewport with a random initial force applied to its center of mass.

```
s (list): The state. Attributes:
            s[0] is the horizontal coordinate
            s[1] is the vertical coordinate
            s[2] is the horizontal speed
            s[3] is the vertical speed
            s[4] is the angle
            s[5] is the angular speed
            s[6] 1 if first leg has contact, else 0
            s[7] 1 if second leg has contact, else 0
```

## Rewards
1. Reward for moving from the top of the screen to the landing pad and coming to rest is about 100-140 points.
2. If the lander crashes, it receives an additional -100 points.
3. If it comes to rest, it receives an additional +100 points.
4. Each leg with ground contact is +10 points.
5. Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame.
6. Solved is 200 points.
## Objective

Create an agent which achieves maximum reward.

## Algorithm

## MDP

Issues:
1. Environment is too complex to understand
2. States like angular speed, velocity etc are difficult to understand and make heuristics

Understanding:
1. Heuristic is probably trying to move the lander towards the angle it makes with center.
2. There is probably no uncertainity.

Approach:
- [x] Start with a random agent to see how it behaves.
- [ ] MDP?
  - MDP will not be suitable as it will take ages to come up with the MDP model parameters i.e. the transition probs.
- [ ] Q Learning
  - https://courses.cs.washington.edu/courses/csep573/22wi/project3/index.html
  - This can be suitable for the discrete actions.
  - We can learn from experience and exploit actinos which have high chances of success while exploring slightly.
  - Problems
    -  the state space is pretty large (find the exact value) and it might again take more time to learn properly.
    -  We need to discretize the state space. This needs knowledge of max and minimum values. This will require deep understanding of state feature ranges and distributions.
  - Try changing the parameters, 
    - discount $\gamma$
    - epsilon $\epsilon$
    - learning rage $\alpha$
- [ ] Approximate Q Learning
  - [ ] This will be more suitable than Q learning as the state variables are continuous.
  - [ ] Will help with generalization
  - [ ] Prolem : need to find out features for $Q(s,a)$
    - [ ] may be i can have 4 functions for 4 separate actions. Use perceptron type logic to find the highest value out of them
    - [ ] https://www.cs.swarthmore.edu/~bryce/cs63/s16/slides/3-25_approximate_Q-learning.pdf
- [ ] Deep Q learning
  - [ ] Better than approx Q learning