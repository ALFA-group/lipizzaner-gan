
mode1 = 'random'
mode2 = 'iterative'
algorithm = 'optimize-random-search'
output_path = './output-potimization/'
generations = 10
for size in range(5,10):
    for i in range (0,10):
        text = 'time CUDA_VISIBLE_DEVICES=1  python main.py {} -f configuration/quickstart/mnist.yml -o {}/output-{}-{}-{:03d}.txt -e {} --generations {} > screen-{}-{}-{:03d}.txt'.format(algorithm, output_path, algorithm, size, i, size, generations,  algorithm, size, i)
        print(text)

# for size in range(4, 10):
#     for i in range(0, 5):
#         text = 'time CUDA_VISIBLE_DEVICES=1  python main.py optimize-greedy -f configuration/quickstart/mnist.yml -o output-{}-{}-{:03d}.txt -e {} --mode {} > screen-{}-{}-{:03d}.txt'.format(mode2, size, i, size, mode2, mode2, size, i)
#         print(text)