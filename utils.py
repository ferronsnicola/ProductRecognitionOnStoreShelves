from sklearn.cluster import DBSCAN
import cv2
import numpy as np
import maincolor
import matplotlib.pyplot as plt

H_RESIZE = 30 # altezza delle immagini che verranno ridotte (solo per un calcolo dei colori dominanti, che su img grandi è pesante)


def detect(models, scene): # restituisce una lista e una lista di liste contenenti tuple KeyPoint-SiftDescriptor, una per ogni modello e una per la scena
	detector = cv2.xfeatures2d.SIFT_create()
	kp_d = [] # lista in cui ogni elemento sarà una lista di tuple [keypoint, descriptor] relativi ad un modello
	for i in range(len(models)):
		kp_d.append(detector.detectAndCompute(models[i], None))
	skp_d = detector.detectAndCompute(scene, None) # lista di tuple [kp, descr]
	return kp_d, skp_d


def compute_matches(models, scene, kp_d, skp_d): # cerca i match di tutti i modelli nella scena
	index_params = dict(algorithm=1, table_number=10, key_size=20, multi_probe_level=2)  # settaggi visti in lab
	#index_params = dict(algorithm=1, table_number=6, key_size=12, multi_probe_level=1)  # settaggi consigliati dalla doc di opencv
	#index_params = dict(algorithm=1, trees=5) # settaggio consigliato dalla doc di opencv per sift
	search_params = dict(checks=50)
	matcher = cv2.FlannBasedMatcher(index_params, search_params)
	result = []
	for i in range(len(models)):
		match = matcher.knnMatch(skp_d[1], kp_d[i][1], 2)  # trovo i match a partire dalla scena (che se contiene 2 volte lo stesso soggetto scazza facendo il contrario)

		thresh = 0.7
		good_match = [] # match forti rispetto ad altri
		for m, n in match:  # seleziono secondo il criterio del rapporto tra distanze
			if m.distance < thresh * n.distance:
				good_match.append(m)
		
		kp_matches = [] # lista di corrispondenze tra kp del modello e di una scena che hanno fatto "good_match"
		for match in good_match:  # per ogni match mi salvo i punti che lo hanno generato
			kp_matches.append((kp_d[i][0][match.trainIdx], skp_d[0][match.queryIdx]))
		
		# per disegnare i kp buoni sulle immagini, debug only
		#for match in kp_matches:
		#	cv2.circle(models[i], (int(match[0].pt[0]), int(match[0].pt[1])), 1, (0,255,0), 2, lineType=cv2.LINE_AA)
		#	cv2.circle(scene, (int(match[1].pt[0]), int(match[1].pt[1])), 1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
		
		result.extend(sift_ght(kp_matches, models, i, scene))
	
	return result # lista di tuple (BB, centroid[point, weight, kp], indice_modello)


def sift_ght(kp_matches, models, index, scene): # restituisce una lista di oggetti (bounding_box,centro di hough,indice del modello)
												# trovati di uno stesso modello nella scena
	### fase offline
	# cerco il reference point come baricentro NEL MODELLO
	ref = kp_matches_barycenter(kp_matches)
	#cv2.circle(models[index], ref, 2, (0, 0, 255), 2, cv2.LINE_AA)  # lo disegno in rosso nel modello
	
	# ho il reference point, ora devo calcolare il vettore che punta al centro da ogni punto
	model_vectors = star_model(kp_matches, ref) # ogni vettore e' fatto da intensita' e angolo
	
	### fase online
	# ora ho tutti i vettori normalizzati e con il loro angolo (assumendo che sia invariante), provo a far votare nella scena
	kpm_votes = voting(kp_matches, model_vectors, scene) # oltre ai punti di voto, salvo anche i pt che l'hanno generato per avere una lista di punti che fanno veramente match
	
	# ora ho tutti i voti, devo analizzarli in modo efficace
	centroids = dbscan_voting_analysis(kpm_votes)
	
	# modifico i pesi secondo il criterio specificato nel commento della funzione
	centroids = weights_tune(centroids)
	
	result = [] # lista di tuple di oggetti trovati relativi ad un modello, contenenti (BB, dati/voti, modello)
	for i in range(len(centroids)):
		#bounding_box = homography(scene, centroids[i][2], models[index]) # teoricamente piu corretto, ma spesso da risultati poco consistenti, BB troppo storti e lancia exception
		bounding_box = get_bounding_box(scene, centroids[i], models[index], ref)
		#if bounding_box is not None: # serve se fallisce l'homography, ma usando la get_bounding_box non serve
		result.append((bounding_box, centroids[i], index))
		
	return result


def weights_tune(centroids):
	# ora ho la posizione di ogni probabile oggetto trovato nella scena, con associato il numero di voti, spesso succede che oggetti
	# simili (ma diversi) vengano rilevati dallo stesso modello e risultino piu forti del vero modello (spesso perche poco risoluto), ma piu deboli rispetto ad un oggetto che
	# effettivamente corrisponde al modello trovato, quindi se troviamo un rapporto di 1.5 tra oggetti che rispondono allo stesso modello
	# allora penalizziamo quello piu debole di un ulteriore 2x per dare vantaggio al piu debole trovato col modello a bassa risoluzione
	centroids = sorted(centroids, key=lambda centroid: centroid[1], reverse=True)
	
	for i in range(1, len(centroids)):
		if 1.5 * centroids[i][1] < centroids[0][1]:
			centroids[i] = (centroids[i][0], centroids[i][1] / 2, centroids[i][2])
	return centroids


def get_bounding_box(scene, centroid, model, ref): # funzione alternativa ad un omografia, piu robusta ma meno teorica
	s_ref = centroid[0] # punto risultante dal voting relativo al modello passato
	h,w,_ = model.shape # altezza e larghezza del modello, serve per sapere trovare il centro del bb rispetto al centroide
	sh, sw, _ = scene.shape # altezza e larghezza della scena, serve per non avere BB che fuoriescono dall'img
	
	scale_sum = 0
	for i in range(len(centroid[2])):
		scale_sum += centroid[2][i][0].size / centroid[2][i][1].size
	avg_scale = scale_sum / len(centroid[2]) # valore medio delle scale dei punti che hanno votato "bene"
	
	dmy = ref[1] - h/2 # distanza dal centroide al centro lungo la y
	dmx = ref[0] - w/2 # distanza dal centroide al centro lungo la x
	
	m_angle = np.inf if dmx == 0 else dmy / dmx # coefficiente angolare del vettore che porta dal centroide al centro
	plus = 0 if (dmy < 0 and dmx < 0) or (dmy > 0 and dmx < 0) else 180  # serve perche facendo solo l'arctan perdo informazioni sul verso del vettore
	v_angle = np.arctan(m_angle) * 180 / np.pi + plus  # in gradi
	v_norm = distance(ref, (w/2, h/2)) / avg_scale
	
	c = (int(s_ref[0] + v_norm * np.cos(v_angle * np.pi / 180.0)), int(s_ref[1] + v_norm * np.sin(v_angle * np.pi / 180.0)))
	
	rect = [] # rettangolo definito dai 4 punti, sempre ordinati nello stesso modo
	rect.append((int(c[0]-w/(2*avg_scale)) if 0 <= c[0]-w/(2*avg_scale) else 0, int(c[1]-h/(2*avg_scale)) if 0 <= c[1]-h/(2*avg_scale) else 0))
	rect.append((int(c[0]+w/(2*avg_scale)) if c[0]+w/(2*avg_scale) <= sw else sw, int(c[1]-h/(2*avg_scale)) if 0 <= c[1]-h/(2*avg_scale) else 0))
	rect.append((int(c[0]+w/(2*avg_scale)) if c[0]+w/(2*avg_scale) <= sw else sw, int(c[1]+h/(2*avg_scale)) if c[1]-h/(2*avg_scale) <= sh else sh))
	rect.append((int(c[0]-w/(2*avg_scale)) if 0 <= c[0]-w/(2*avg_scale) else 0, int(c[1]+h/(2*avg_scale)) if c[1]-h/(2*avg_scale) <= sh else sh))
	rect = np.array([rect])
	
	return rect


def homography(scene, true_matches, model): # chiamata per ogni centroide, trova l'omografia e il BB da disegnare nella scena
	src_pts = np.float32([true_matches[i][0].pt for i in range(len(true_matches))]).reshape(-1, 1, 2)
	dst_pts = np.float32([true_matches[i][1].pt for i in range(len(true_matches))]).reshape(-1, 1, 2)
	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
	h, w, _ = model.shape
	if M is None:
		return None
	pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
	dst = cv2.perspectiveTransform(pts, M)
	if homography_test(dst)==True:
		return dst
	else:
		return None
	

def homography_test(rect): # test per verificare che il rettangolo non abbia forme troppo sbagliate
	test = True
	psusx = rect[0][0]
	pgiusx = rect[1][0]
	pgiudx = rect[2][0]
	psudx = rect[3][0]
	
	if distance(psusx, psudx) / distance(pgiudx, pgiusx) < 0.6 or distance(psusx, psudx) / distance(pgiudx,pgiusx) > 1.4:
		test = False
	if distance(psusx, pgiusx) / distance(pgiudx, psudx) < 0.6 or distance(psusx, pgiusx) / distance(pgiudx,psudx) > 1.4:
		test = False
	ff = distance(psusx, pgiusx) / distance(psusx, psudx)
	if ff > 2 or ff < 0.5:
		test = False
	return test

def star_model(kp_matches, ref): # per ogni kp calcolo il vettore che lo porta in reg (baricentro dei kp)
	vectors = []
	for m in kp_matches:
		v_norm = distance(m[0].pt, ref) / m[0].size
		dmy = m[0].pt[1] - ref[1]
		dmx = m[0].pt[0] - ref[0]
		m_angle = dmy / dmx
		plus = 0 if (dmy < 0 and dmx < 0) or (
					dmy > 0 and dmx < 0) else 180  # serve perche facendo solo l'arctan perdo informazioni sul verso del vettore
		v_angle = (np.arctan(m_angle) * 180 / np.pi + plus)  - m[0].angle  # deg
		vectors.append((v_norm, v_angle))
	return vectors

def voting(kp_matches, model_vectors, scene): # salvo, per ogni kp, il voto che si ottiene "applicandogli" il vettore relativo
	votes = []
	for i in range(len(kp_matches)):
		skp = kp_matches[i][1]
		mkp = kp_matches[i][0]
		v = model_vectors[i][0] * skp.size
		angle = model_vectors[i][1]  + skp.angle
		c = (skp.pt[0] + v * np.cos(angle * np.pi / 180.0), skp.pt[1] + v * np.sin(angle * np.pi / 180.0), mkp.size/skp.size)
		#cv2.circle(scene, c, 1, (0, 0, 255), 1, cv2.LINE_AA)  # per disegnare tutti i voti
		votes.append((c, kp_matches[i]))
	votes = np.array(votes)
	return votes

def dbscan_voting_analysis(kpm_votes): # analisi dei voti secondo l'algoritmo di clustering density based
	temp = kpm_votes[:,0]
	temp = np.array(temp)
	
	votes = [] # senza fare questo ciclo, dbscan non accetta la collection di voti che gli passo
	for i in range(len(temp)):
		votes.append(np.array(temp[i]))
	votes = np.array(votes)

	db = DBSCAN(eps=15, min_samples=3).fit(votes)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True

	n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
	
	# trovo centro e peso di ogni cluster (oggetto trovato)
	centroids = []
	for i in range(n_clusters_):
		class_member_mask = (db.labels_ == i)
		good_votes = votes[class_member_mask & core_samples_mask] # voti vincitori
		good_votes_matching_kp = []
		for j in range(len(votes)):
			for k in range(len(good_votes)):
				if good_votes[k][0] == kpm_votes[j][0][0] and good_votes[k][1] == kpm_votes[j][0][1]: # salvo tutti i kp che hanno votato bene
					good_votes_matching_kp.append(kpm_votes[j][1])
					continue
		c = np.average(votes[db.labels_ == i], axis=0) # baricentro dei voti
		centroids.append((c, len(votes[db.labels_ == i]), good_votes_matching_kp)) # per ogni centroide salvo il punto, il numero di voti ricevuti (funge da peso),
																					#  e i kp che hanno votato i punti facenti parte del cluster
	
	return centroids


def filter_repetition(list, scene, models, models_main_colors): # list = lista di tuple (BB, centroid[point, weight, kp], indice_modello)
	result = [] # elementi di list che sono stati "approvati"
	rejected_index = [] # mi salvo gli indici che sono già stati scartati, rendendo più efficiente l'algoritmo
	max_weight = 0 # serve per determinare il peso maggiore in una scena, se un oggetto verra' trovato con un peso molto minore lo scarto
					# (mi permette di non modificare parametri di voting tra scene con oggetti grandi e scene con oggetti piccoli)
	for i in range(len(list)):
		if i in rejected_index:
			continue
		test = True # se trovero un rettangolo migliore lo mettero a false (e non lo aggiungero al risultato)
		rect = list[i][0]
#		if rect is not None: # solo se si usa l'omografia, con la funzione get_bounding_box non serve
		position = barycenter(rect)
		weight = list[i][1][1]
		if weight > max_weight:
			max_weight = weight
		for j in range(len(list)):
			if i in rejected_index or i==j:
				continue
			if cv2.pointPolygonTest(list[j][0],position,False) == 1: # se il baricentro del box in analisi e' contenuto in un altro box
				s_img = get_resized_img(scene[rect[0][0][1]:rect[0][2][1], rect[0][0][0]:rect[0][1][0]]) # porzione della scena che contiene l'oggetto riconosciuto, ridimensionato perchè serve solo per determinarne il colore
				scene_object_maincolors = maincolor.get_main_colors(s_img, 2) # calcolo i due colori principali
				color_distance = maincolor.main_colors_difference(scene_object_maincolors, models_main_colors[list[i][2]]) # calcolo le distanze tra i colori
				color_distance2 = maincolor.main_colors_difference(scene_object_maincolors, models_main_colors[list[j][2]])

				if list[j][1][1] * (color_distance/color_distance2) > weight: # al peso dovuto ai voti di hough aggiungo un fattore dovuto alla distanza
					test = False
					rejected_index.append(i)
					break
				else:
					rejected_index.append(j)
		if test:
			result.append(list[i])
				
	return result, max_weight

def get_resized_img(img):
	h, w, _ = img.shape
	if h > H_RESIZE:
		scale = H_RESIZE / h
		img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR);
	return img

def get_result(list, scene, max_weight): # serve per validare i risultato senza che ci siano troppe differenze in termini di voti tra il piu votato e il meno, mi permette di non variare i parametri di voting
	for i in range(len(list)):
		if list[i][1][1] > max_weight/15:
			cv2.polylines(scene, np.int32([list[i][0]]), True, (0,255,0), 2, cv2.LINE_AA)
			cv2.putText(scene, str(list[i][2]), barycenter(list[i][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
			print("modello: " + str(list[i][2]) +", position: " + str(barycenter(list[i][0])) + ", dimensions: " + str(distance(list[i][0][0][0], list[i][0][0][1])) + "x" + str(distance(list[i][0][0][1], list[i][0][0][2])))


def distance(p0, p1):
	return np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)

def kp_matches_barycenter(kp_matches): # utile per trovare il bc in una lista di mie strutture dati
	x = 0
	y = 0
	for m in kp_matches:
		x += m[0].pt[0]
		y += m[0].pt[1]
	result = (int(x / len(kp_matches)), int(y / len(kp_matches)))
	return result

def barycenter(points):
	x=0
	y=0
	for i in range(len(points[0])):
		x+=points[0][i][0]
		y+=points[0][i][1]
	result = (int(x/len(points[0])), int(y/len(points[0])))
	return result
