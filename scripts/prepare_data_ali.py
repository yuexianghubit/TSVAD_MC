import glob, tqdm, os, textgrid, soundfile, copy, json, argparse, numpy, torch, wave
from collections import defaultdict

class Segment(object):
	def __init__(self, uttid, spkr, stime, etime, text, name):
		self.uttid = uttid
		self.spkr = spkr
		self.stime = round(stime, 2)
		self.etime = round(etime, 2)
		self.text = text
		self.name = name

	def change_stime(self, time):
		self.stime = time

	def change_etime(self, time):
		self.etime = time

def remove_overlap(aa, bb):
	# Sort the intervals in both lists based on their start time
	a = aa.copy()
	b = bb.copy()
	a.sort()
	b.sort()

	# Initialize the new list of intervals
	result = []

	# Initialize variables to keep track of the current interval in list a and the remaining intervals in list b
	i = 0
	j = 0

	# Iterate through the intervals in list a
	while i < len(a):
		# If there are no more intervals in list b or the current interval in list a does not overlap with the current interval in list b, add it to the result and move on to the next interval in list a
		if j == len(b) or a[i][1] <= b[j][0]:
			result.append(a[i])
			i += 1
		# If the current interval in list a completely overlaps with the current interval in list b, skip it and move on to the next interval in list a
		elif a[i][0] >= b[j][0] and a[i][1] <= b[j][1]:
			i += 1
		# If the current interval in list a partially overlaps with the current interval in list b, add the non-overlapping part to the result and move on to the next interval in list a
		elif a[i][0] < b[j][1] and a[i][1] > b[j][0]:
			if a[i][0] < b[j][0]:
				result.append([a[i][0], b[j][0]])
			a[i][0] = b[j][1]
		# If the current interval in list a starts after the current interval in list b, move on to the next interval in list b
		elif a[i][0] >= b[j][1]:
			j += 1

	# Return the new list of intervals
	return result

def get_args():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--data_path', help='the path for the alimeeting')
	parser.add_argument('--type', help='Eval or Train')
	parser.add_argument('--length_embedding', type=float, default=6, help='length of embeddings, seconds')
	parser.add_argument('--step_embedding', type=float, default=1, help='step of embeddings, seconds')
	parser.add_argument('--batch_size', type=int, default=96, help='batch size')

	args = parser.parse_args()
	args.path = os.path.join(args.data_path, '%s_Ali/%s_Ali_far'%(args.type,args.type))
	args.path_wav = os.path.join(args.path, 'audio_dir')
	args.path_grid = os.path.join(args.path, 'textgrid_dir')
	args.target_wav = os.path.join(args.path, 'target_audio_mc')
	args.target_embedding = os.path.join(args.path, 'target_embedding')
	args.out_text = os.path.join(args.path, '%s.json'%(args.type))
	return args

def main():
	args = get_args()
	text_grids = glob.glob(args.path_grid + '/*')
	outs = open(args.out_text, "w")
	for text_grid in tqdm.tqdm(text_grids):
		tg = textgrid.TextGrid.fromFile(text_grid)
		segments = []
		spk = {}
		num_spk = 1
		uttid = text_grid.split('/')[-1][:-9]
		for i in range(tg.__len__()):
			for j in range(tg[i].__len__()):
				if tg[i][j].mark:
					if tg[i].name not in spk:
						spk[tg[i].name] = num_spk
						num_spk += 1
					segments.append(Segment(
							uttid,
							spk[tg[i].name],
							tg[i][j].minTime,
							tg[i][j].maxTime,
							tg[i][j].mark.strip(),
							tg[i].name
						)
					)
		segments = sorted(segments, key=lambda x: x.spkr)

		intervals = defaultdict(list)
		new_intervals = defaultdict(list)

		dic = defaultdict()
		# Summary the intervals for all speakers
		for i in range(len(segments)):
			interval = [segments[i].stime, segments[i].etime]
			intervals[segments[i].spkr].append(interval)
			dic[str(segments[i].uttid) + '_' + str(segments[i].spkr)] = segments[i].name.split('_')[-1]

		# Remove the overlapped speeech    
		for key in intervals:
			new_interval = intervals[key]
			for o_key in intervals:
				if o_key != key:                
					new_interval = remove_overlap(copy.deepcopy(new_interval), copy.deepcopy(intervals[o_key]))
			new_intervals[key] = new_interval

		wav_file = glob.glob(os.path.join(args.path_wav, uttid) + '*.wav')[0]
		orig_audio_mc, _ = soundfile.read(wav_file)
		orig_audio = orig_audio_mc[:,0]
		length = len(orig_audio)

		# # Cut and save the clean speech part
		id_full = wav_file.split('/')[-1][:-4]
		room_id = id_full[:11]
		for key in new_intervals:
			output_dir = os.path.join(args.target_wav, id_full)
			os.makedirs(output_dir, exist_ok = True)
			output_wav = os.path.join(output_dir, str(key) + '.wav')
			new_audio = []
			#labels = [0] * int(length / 16000 * 25) # 40ms, one label
			for interval in new_intervals[key]:
				s, e = interval
				#for i in range(int(s * 25), min(int(e * 25) + 1, len(labels))):
				#	labels[i] = 1
				s *= 16000
				e *= 16000
				new_audio.extend(orig_audio[int(s):int(e)])
			soundfile.write(output_wav, new_audio, 16000)
		output_wav = os.path.join(output_dir, 'all.wav')
		soundfile.write(output_wav, orig_audio_mc, 16000)

		# Save the labels
		for key in intervals:
			labels = [0] * int(length / 16000 * 25) # 40ms, one label
			labels_piw = [0] * int(length / 16000 * 25) 
			for interval in intervals[key]:
				s, e = interval
				for i in range(int(s * 25), min(int(e * 25) + 1, len(labels))):
					# diarization label, 0: inactive, 1: active
					labels[i] = 1
					# position in word (PIW) label, 0: silence, 1: singleton, 2: begin of speech, 3: internal of speech, 4: end of speech, 5: begin of silence, 6: end of silence
					if (e - s) * 25 <= 10: # we find speaking one Chinese word about 0.27-0.36s based on "R0005_M0041.TextGrid"
						labels_piw[i] = 1 # if the segment is less than 10 frames, tag "singleton"
					else:
						if i > int(s * 25) and i < min(int(e * 25) + 1, len(labels)) - 1:
							labels_piw[i] = 3 # tag "internal of speech" in PIW
						if i == int(s * 25):
							labels_piw[i] = 2 # tag "begin of speech"
							if i - 1 >= 0: labels_piw[i] = 6 # tag "end of silence"
						if i == min(int(e * 25) + 1, len(labels)) - 1:
							labels_piw[i] = 4 # tag "end of speech"
							if i + 1 <= len(labels): labels_piw[i] = 5 # tag "begin of silence"

			room_speaker_id = room_id + '_' + str(key)
			speaker_id = dic[room_speaker_id]

			res = {'filename':id_full, 'speaker_key':key, 'speaker_id':speaker_id, 'labels':labels, 'labels_piw':labels_piw}
			json.dump(res, outs)
			outs.write('\n')


if __name__ == '__main__':
	main()
