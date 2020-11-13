generations = 10000
n_samples = 1000
output_path = './output-potimization/'

algorithm_ga = 'optimize-ga'
pop_size = 100
mutation_prob = 0.1
crossover_prob = 0.5
mutation_type = 'combined-mutation'

algorithm_greddy = 'optimize-greedy'
mode1 = 'greddy-random'
mode2 = 'greddy-iterative'

algorithm_random = 'optimize-random-search'
j=0
for size in [200]:
    for i in range (430, 460):
        for pcr in [0.75]:
        #for pm in [0.01, 0.05, 0.1]:
            #for pcr in [0.25, 0.50, 0.75]:
            for pm in [0.1]:
                generations = 10000
                j+=1
                mutation_prob = pm
                crossover_prob = pcr
                text = 'time CUDA_VISIBLE_DEVICES={}  python main.py {} -f configuration/quickstart/mnist.yml -o {}/output-{}-{}-{:03d}-{}-{}-{}.txt -e {} --generations {} --n_samples {} --population_size {} -mp {} -cp {} > {}/screen-new-{}-{}-{:03d}-{}-{}-{}.txt'.format(
                    j%2, algorithm_ga,
                    output_path, algorithm_ga, size, i, mutation_prob, crossover_prob, mutation_type, size, generations, n_samples,
                    pop_size, mutation_prob, crossover_prob, output_path,
                    algorithm_ga, size,
                    i, mutation_prob, crossover_prob, mutation_type)

                if j%5==0:
                    print(text)
                    print('sleep 60')
                else:
                    print(text + '  &')

import sys
sys.exit(0)
# for size in [8]:
#     for i in range (110, 120):
#         #for pcr in [0.25]:
#         for pm in [0.01, 0.05, 0.1]:
#             for pcr in [0.25, 0.50, 0.75]:
#             #for pm in [0.1]:
#                 generations = 10000
#                 j+=1
#                 mutation_prob = pm
#                 crossover_prob = pcr
#                 text = 'time CUDA_VISIBLE_DEVICES={}  python main.py {} -f configuration/quickstart/mnist.yml -o {}/output-{}-{}-{:03d}-{}-{}-{}.txt -e {} --generations {} --n_samples {} --population_size {} -mp {} -cp {} > {}/screen-{}-{}-{:03d}-{}-{}-{}.txt'.format(
#                     j%2, algorithm_ga,
#                     output_path, algorithm_ga, size, i, mutation_prob, crossover_prob, mutation_type, size, generations, n_samples,
#                     pop_size, mutation_prob, crossover_prob, output_path,
#                     algorithm_ga, size,
#                     i, mutation_prob, crossover_prob, mutation_type)
#
#                 if j%20==0:
#                     print(text)
#                     print('sleep 60')
#                 else:
#                     print(text + '  &')
#
# for size in [8]:
#     for i in range(130, 150):
#         # for pcr in [0.25]:
#         for pm in [0.01, 0.05, 0.1]:
#             for pcr in [0.25, 0.50, 0.75]:
#                 # for pm in [0.1]:
#                 generations = 10000
#                 j += 1
#                 mutation_prob = pm
#                 crossover_prob = pcr
#                 text = 'time CUDA_VISIBLE_DEVICES={}  python main.py {} -f configuration/quickstart/mnist.yml -o {}/output-{}-{}-{:03d}-{}-{}-{}.txt -e {} --generations {} --n_samples {} --population_size {} -mp {} -cp {} > {}/screen-{}-{}-{:03d}-{}-{}-{}.txt'.format(
#                     j % 2, algorithm_ga,
#                     output_path, algorithm_ga, size, i, mutation_prob, crossover_prob, mutation_type, size, generations,
#                     n_samples,
#                     pop_size, mutation_prob, crossover_prob, output_path,
#                     algorithm_ga, size,
#                     i, mutation_prob, crossover_prob, mutation_type)
#
#                 if j % 20 == 0:
#                     print(text)
#                     print('sleep 60')
#                 else:
#                     print(text + '  &')
#
# #

#
# import sys
# sys.exit(0)
#
#
#
j=0
for size in [9, 10]:
    for i in range (130,145):
        text = 'time CUDA_VISIBLE_DEVICES={}  python main.py {} -f configuration/quickstart/mnist.yml -o {}/output-{}-{}-{:03d}.txt -e {} --generations {} --n_samples {} > {}/screen-{}-{}-{:03d}.txt'.format(
            j%2, algorithm_random, output_path, algorithm_random, size, i, size, generations,  n_samples, output_path, algorithm_random, size, i)
        j+=1
        if j % 20 == 0:
            print(text)
            print('sleep 60')
        else:
            print(text + '  &')

# sys.exit(0)

# for size in [8, 3, 4]:
#     for i in range (0,30):
#         text = 'time CUDA_VISIBLE_DEVICES=0  python main.py {} -f configuration/quickstart/mnist.yml -o {}/output-{}-{}-{:03d}.txt -e {} --generations {} --n_samples {} > {}/screen-{}-{}-{:03d}.txt'.format(
#             algorithm_random, output_path, algorithm_random, size, i, size, generations,  n_samples, output_path, algorithm_random, size, i)
#         j+=1
#         if j % 20 == 0:
#             print(text)
#             print('sleep 60')
#         else:
#             print(text + '  &')
#

#
#         # text = 'time CUDA_VISIBLE_DEVICES=0  python main.py {} -f configuration/quickstart/mnist.yml -o {}/output-{}-{}-{}-{:03d}.txt -e {} --n_samples {} --mode {} > screen-{}-{}-{:03d}.txt'.format(
#         #     algorithm_greddy, output_path, algorithm_greddy, mode1, size, i, size, n_samples, mode1, algorithm_greddy, size,
#         #     i)
#         # print(text + ' ')
#         # text = 'time CUDA_VISIBLE_DEVICES=0  python main.py {} -f configuration/quickstart/mnist.yml -o {}/output-{}-{}-{}-{:03d}.txt -e {} --n_samples {} --mode {} > screen-{}-{}-{:03d}.txt'.format(
#         #     algorithm_greddy, output_path, algorithm_greddy, mode2, size, i, size, n_samples, mode2, algorithm_greddy, size,
#         #     i)
#         # print(text + ' ')
#         #
#         # text = 'time CUDA_VISIBLE_DEVICES=0  python main.py {} -f configuration/quickstart/mnist.yml -o {}/output-{}-{}-{:03d}.txt -e {} --generations {} --n_samples {} --population_size {} -mp {} -cp {} > screen-{}-{}-{:03d}.txt'.format(
#         #     algorithm_ga, output_path, algorithm_ga, size, i, size, generations, n_samples,
#         #     pop_size, mutation_prob, crossover_prob,
#         #     algorithm_greddy, size,
#         #     i)
#         # print(text + ' ')
# j = 0
# for size in [5]:
#     for i in range(150, 180):
#         j += 1
#         text = 'time CUDA_VISIBLE_DEVICES={}  python main.py optimize-greedy -f configuration/quickstart/mnist.yml -o {}/output-{}-{}-{:03d}.txt -e {} --mode {} --n_samples {}  > {}/screen-{}-{}-{:03d}.txt'.format(
#             j%2,
#             output_path, mode2, size, i, size, mode2, n_samples, output_path, mode2, size, i)
#
#         # text = 'time CUDA_VISIBLE_DEVICES={}  python main.py optimize-greedy -f configuration/quickstart/mnist.yml -o {}/output-{}-{}-{:03d}.txt -e {} --mode {} --n_samples {} > {}/screen-{}-{}-{:03d}.txt'.format(
#         #     j%2,
#         #     output_path, mode1, size, i, size, mode1, n_samples, output_path, mode1, size, i)
#
#         if j % 15 == 0:
#             print(text)
#             print('sleep 60')
#         else:
#             print(text + '  &')
#         #
#         # j += 1
#         # text = 'time CUDA_VISIBLE_DEVICES={}  python main.py optimize-greedy -f configuration/quickstart/mnist.yml -o {}/output-{}-{}-{:03d}.txt -e {} --mode {} --n_samples {} > {}/screen-{}-{}-{:03d}.txt'.format(
#         #     0%2,
#         #     output_path, mode1, size, i, size, mode1, n_samples, output_path, mode1, size, i)
#         #
#         # if j % 20 == 0:
#         #     print(text)
#         #     print('sleep 60')
#         # else:
#         #     print(text + '  &')
