import time
import maincolor
import cv2
import utils

path = "imgs/"

models = []
for i in range(27):
	img = cv2.imread(path + "models/" + str(i) + ".jpg")
	models.append(img)

models_main_colors = []
for i in range(len(models)):
	m_img = utils.get_resized_img(models[i])
	models_main_colors.append(maincolor.get_main_colors(m_img, 2))

scenes = []
for i in range(1,6):
#	scenes.append(cv2.imread(path + "scenes/e" + str(i) + ".png"))
	scenes.append(cv2.imread(path + "scenes/h" + str(i) + ".jpg"))
#	scenes.append(cv2.imread(path + "scenes/m" + str(i) + ".png"))


micros = time.time()

for i in range(len(scenes)):
	print ("working on scene " + str(i))
	kp_d, skp_d = utils.detect(models, scenes[i])
	print("computing matches")
	all_recognized_objects = utils.compute_matches(models, scenes[i], kp_d, skp_d)
	print("filtering")
	true_rect, max_weight = utils.filter_repetition(all_recognized_objects, scenes[i], models, models_main_colors)
	print("ending")
	utils.get_result(true_rect, scenes[i], max_weight)
	
print(time.time() - micros)
	
##############
### OUTPUT ###
##############

for i in range(len(scenes)):
	cv2.imshow("scene" + str(i+1), scenes[i])

cv2.waitKey(0)
