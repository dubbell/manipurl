import torch
import torch.nn as nn

from manipurl.models.clip import project_image, project_text


class FuRLShaper:

    def __init__(self, task_description, lr = 1e-4):
        self.image_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64)).cuda()
        
        self.text_head = nn.Sequential  (
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64)).cuda()
        
        self.clip_task_embedding = project_text(task_description)


    def get_reward(self, image):
        clip_image_embedding = project_image(image)
        
        proj_image = self.image_head.forward(clip_image_embedding)
        proj_text = self.text_head.forward(self.clip_task_embedding)

        return nn.functional.cosine_similarity(proj_image, proj_text)
    

