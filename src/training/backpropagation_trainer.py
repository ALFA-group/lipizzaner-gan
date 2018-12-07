import logging

import losswise
import numpy as np
import torch

from helpers.configuration_container import ConfigurationContainer
from helpers.pytorch_helpers import to_pytorch_variable
from training.nn_trainer import NeuralNetworkTrainer


class BackpropagationTrainer(NeuralNetworkTrainer):
    _logger = logging.getLogger(__name__)

    def initialize_populations(self):
        return None, None

    def train(self, n_iterations, stop_event=None):

        cc = ConfigurationContainer.instance()
        session = None
        graph_loss = None
        graph_step_size = None

        generator = self.network_factory.create_generator()
        discriminator = self.network_factory.create_discriminator()

        g_optimizer = torch.optim.Adam(generator.net.parameters(), lr=0.0003)
        d_optimizer = torch.optim.Adam(discriminator.net.parameters(), lr=0.0003)

        if cc.is_losswise_enabled:
            session = losswise.Session(tag=self.__class__.__name__, max_iter=n_iterations)
            graph_loss = session.graph('loss', kind='min')
            graph_step_size = session.graph('step_size')

        loaded = self.dataloader.load()
        for epoch in range(n_iterations):

            step_sizes_gen = []
            step_sizes_dis = []

            for i, (images, labels) in enumerate(loaded):
                # Store previous parameters for step size computation
                w_gen_previous = generator.parameters
                w_dis_previous = discriminator.parameters

                # Build mini-batch dataset
                batch_size = images.size(0)
                images = to_pytorch_variable(images.view(batch_size, -1))

                # ============= Train the discriminator =============#
                d_loss = discriminator.compute_loss_against(generator, images)[0]

                discriminator.net.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                # =============== Train the generator ===============#
                g_loss, fake_images = generator.compute_loss_against(discriminator, images)

                discriminator.net.zero_grad()
                generator.net.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                step_sizes_gen.append(np.linalg.norm(generator.parameters - w_gen_previous))
                step_sizes_dis.append(np.linalg.norm(discriminator.parameters - w_dis_previous))

                if (i + 1) % 300 == 0:
                    self._logger.info('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, '
                                      'g_loss: %.4f'
                                      % (epoch, n_iterations, i + 1, 600, d_loss.data[0], g_loss.data[0]))

            if graph_loss is not None:
                graph_loss.append(epoch,
                                  {'L(gen(x)) - Backprop': float(g_loss), 'L(disc(x)) - Backprop': float(d_loss)})
                graph_step_size.append(epoch, {
                    'avg_step_size(g(x)) - Backprop': np.mean(step_sizes_gen),
                    'avg_step_size(d(x)) - Backprop': np.mean(step_sizes_dis)})

            # Some datesets (e.g. ImageFolder) do not need shapes
            shape = loaded.dataset.train_data.shape if hasattr(loaded.dataset, 'train_data') else None

            # Save real images once
            if epoch == 0:
                self.dataloader.save_images(images, shape, 'real_images.png')

            z = to_pytorch_variable(torch.randn(min(batch_size, 100), self.network_factory.gen_input_size))
            generated_output = generator.net(z)
            self.dataloader.save_images(generated_output, shape, 'fake_images-%d.png' % (epoch + 1))
            self._logger.info('Epoch [%d/%d], d_loss: %.4f, g_loss: %.4f'
                              % (epoch, n_iterations, d_loss.data[0], g_loss.data[0]))

            #
            # # Save real images once
            # if epoch == 0:
            #     save_images(images, loaded.dataset.train_data.shape, 'real_images.png')
            #
            # # Save sampled images
            # save_images(fake_images, loaded.dataset.train_data.shape, 'fake_images-%d.png' % (epoch + 1))

        if session is not None:
            session.done()

        return (generator, g_loss), (discriminator, d_loss)

