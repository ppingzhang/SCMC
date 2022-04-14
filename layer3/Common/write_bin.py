import struct
from pathlib import Path



def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size

def write_uints(fd, values, fmt=">{:d}I"):
	fd.write(struct.pack(fmt.format(len(values)), *values))


def write_uchars(fd, values, fmt=">{:d}B"):
	fd.write(struct.pack(fmt.format(len(values)), *values))


def read_uints(fd, n, fmt=">{:d}I"):
	sz = struct.calcsize("I")
	return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
	sz = struct.calcsize("B")
	return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
	if len(values) == 0:
		return
	fd.write(struct.pack(fmt.format(len(values)), values))


def read_bytes(fd, n, fmt=">{:d}s"):
	sz = struct.calcsize("s")
	return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def get_header(model_name, metric, quality):
	"""Format header information:
	- 1 byte for model id
	- 4 bits for metric
	- 4 bits for quality param
	"""
	metric = metric_ids[metric]
	code = (metric << 4) | (quality - 1 & 0x0F)
	return model_ids[model_name], code


def parse_header(header):
	"""Read header information from 2 bytes:
	- 1 byte for model id
	- 4 bits for metric
	- 4 bits for quality param
	"""
	model_id, code = header
	quality = (code & 0x0F) + 1
	metric = code >> 4
	return (
		inverse_dict(model_ids)[model_id],
		inverse_dict(metric_ids)[metric],
		quality,
	)


def write_to_bin(out, h=256, w=256, save_path='./out.bin', lmbda=1): #model='model1', metric='mse', quality=255):
	shape = out["shape"]
	#header = get_header(model, metric, quality)

	with Path(save_path).open("wb") as f:
		#print(lmbda, h, w, type(lmbda), ty)

		write_uints(f, [lmbda])
		write_uints(f, (h, w))

		# write shape and number of encoded latents
		write_uints(f, (shape[0], shape[1], len(out["strings"])))

		for s in out["strings"]:
			#print(len(s[0]), len(out["strings"]))
			write_uints(f, (len(s[0]),))
			write_bytes(f, s[0])

	size = filesize(save_path)
	bpp = float(size) * 8 / (h*w)
	return bpp


def write_to_bin1(out, h=256, w=256, save_path='./out.bin', lmbda=1): #model='model1', metric='mse', quality=255):
	shape = out["shape"]
	#header = get_header(model, metric, quality)

	with Path(save_path).open("wb") as f:
		#print(lmbda, h, w, type(lmbda), ty)

		write_uints(f, [lmbda])
		write_uints(f, (h, w))

		# write shape and number of encoded latents
		write_uints(f, (shape[0], shape[1], len(out["strings"])))

		for s in out["strings"]:

			#print(len(s[0]), len(out["strings"]))
			write_uints(f, (len(s[0]),))
			write_bytes(f, s[0])



	#with Path("./out_y.bin").open("wb") as f:
	#	write_bytes(f, out["strings"][0][0])
	#sizey = filesize('out_y.bin')


	#with Path("./out_z.bin").open("wb") as f:
	#	write_bytes(f, out["strings"][1][0])
	#sizez = filesize('out_z.bin')
#
	#print('real_y:{:.5f}'.format(float(sizey) * 8 / (h*w)), 'real_z:{:.5f}'.format(float(sizez) * 8 / (h * w)))


	size = filesize(save_path)
	bpp = float(size) * 8 / (h*w)
	return bpp

def read_from_bin(inputpath):
	with Path(inputpath).open("rb") as f:
		#model, metric, quality = parse_header(read_uchars(f, 2))
		lmbda = read_uints(f, 1)
		original_size = read_uints(f, 2)
		shape = read_uints(f, 2)
		strings = []
		n_strings = read_uints(f, 1)[0]
		for _ in range(n_strings):
			s = read_bytes(f, read_uints(f, 1)[0])
			strings.append([s])

	#print(f"Model: {model:s}, metric: {metric:s}, quality: {quality:d}")
	return strings, original_size, shape,  lmbda



def write_to_bin2(out, h=256, w=256, save_path='./out.bin', lmbda=1): #model='model1', metric='mse', quality=255):


	shape = out["shape"]

	with Path(save_path).open("wb") as f:

		write_uints(f, [lmbda])
		write_uints(f, (h, w))
		#print(shape[0], '===+++++')

		ss0 = shape[0]
		write_uints(f, (ss0[0], ss0[1]))
		ss1 = shape[1]
		write_uints(f, (ss1[0], ss1[1]))
		ss2 = shape[2]
		write_uints(f, (ss2[0], ss2[1]))
		ss3 = shape[3]
		write_uints(f, (ss3[0], ss3[1]))

		write_uints(f, [len(out["strings"])])
		for s in out["strings"]:
			write_uints(f, (len(s[0]),))
			write_bytes(f, s[0])

	size = filesize(save_path)
	bpp = float(size) * 8 / (h*w)
	return bpp


def read_from_bin2(inputpath):
	shape_all= []
	with Path(inputpath).open("rb") as f:
		#model, metric, quality = parse_header(read_uchars(f, 2))
		lmbda = read_uints(f, 1)
		original_size = read_uints(f, 2)

		shape = read_uints(f, 2)
		shape_all.append(shape)
		shape1 = read_uints(f, 2)
		shape_all.append(shape1)
		shape2 = read_uints(f, 2)
		shape_all.append(shape2)
		shape3 = read_uints(f, 2)
		shape_all.append(shape3)

		strings = []
		n_strings = read_uints(f, 1)[0]
		for _ in range(n_strings):
			s = read_bytes(f, read_uints(f, 1)[0])
			strings.append([s])

	#print(f"Model: {model:s}, metric: {metric:s}, quality: {quality:d}")
	return strings, original_size, shape_all, lmbda



def write_to_bin_check(out, h=256, w=256, save_path='./out.bin', lmbda=1): #model='model1', metric='mse', quality=255):
	shape = out["shape"]
	shape1 = out["shape2"]
	#header = get_header(model, metric, quality)

	with Path(save_path).open("wb") as f:
		#print(lmbda, h, w, type(lmbda), ty)

		write_uints(f, [lmbda])
		write_uints(f, (h, w))
		write_uints(f, (shape1[0], shape1[1], shape1[2], shape1[3]))

		write_uints(f, (shape[0], shape[1], len(out["strings"])))

		for s in out["strings"]:

			#print(len(s[0]), len(out["strings"]))
			write_uints(f, (len(s[0]),))
			write_bytes(f, s[0])


def read_from_bin_check(inputpath):
	with Path(inputpath).open("rb") as f:
		#model, metric, quality = parse_header(read_uchars(f, 2))
		lmbda = read_uints(f, 1)
		original_size = read_uints(f, 2)
		shape1 = read_uints(f, 4)
		print(shape1)
		shape = read_uints(f, 2)
		strings = []
		n_strings = read_uints(f, 1)[0]
		for _ in range(n_strings):
			s = read_bytes(f, read_uints(f, 1)[0])
			strings.append([s])

	#print(f"Model: {model:s}, metric: {metric:s}, quality: {quality:d}")
	return strings, original_size, shape, shape1, lmbda

