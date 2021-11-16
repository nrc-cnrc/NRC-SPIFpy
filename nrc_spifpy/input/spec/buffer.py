import numpy as np

from nrc_spifpy.input.spec.image import ImageMetadataProcessor
from nrc_spifpy.input.spec.image import RawImageExtractor
from nrc_spifpy.input.spec.image import RawImageDecoder
from nrc_spifpy.input.spec.image import DecodedImageDecompressor
from nrc_spifpy.input.spec.image import ImageRecordAssembler

from nrc_spifpy.input.spec import housekeeping

DATA_FLAG = (ord('2') << 8) + ord('S')
HK_FLAG = (ord('H') << 8) + ord('K')
MK_FLAG = (ord('M') << 8) + ord('K')

mask_template = np.dtype([
    ("flag","u2"),
    ("timing_word", "int32"),
    ("hz_mask_byte_1", "u2"),
    ("hz_mask_byte_2", "u2"),
    ("hz_mask_byte_3", "u2"),
    ("hz_mask_byte_4", "u2"),
    ("hz_mask_byte_5", "u2"),
    ("hz_mask_byte_6", "u2"),
    ("hz_mask_byte_7", "u2"),
    ("hz_mask_byte_8", "u2"),
    ("v_mask_byte_1", "u2"),
    ("v_mask_byte_2", "u2"),
    ("v_mask_byte_3", "u2"),
    ("v_mask_byte_4", "u2"),
    ("v_mask_byte_5", "u2"),
    ("v_mask_byte_6", "u2"),
    ("v_mask_byte_7", "u2"),
    ("v_mask_byte_8", "u2"),
    ("timing_word_start", "int32"),
    ("timing_word_end", "int32")
])

class Buffer:

    def __init__(self, buffer_id, buffer_ts, buffer):
        self.buffer_id = buffer_id
        self.buffer_sec = self.get_buffer_sec(buffer_ts)
        self.buffer_ns = self.get_buffer_ns(buffer_ts)
        self.buffer = buffer

        self.metadata_processor = ImageMetadataProcessor()
        self.raw_image_extractor = RawImageExtractor()
        self.raw_image_decoder = RawImageDecoder()
        self.decoded_image_decompressor = DecodedImageDecompressor()
        self.image_record_assembler = ImageRecordAssembler()

        self.metadata_containers = []
        self.raw_image_containers = []
        self.image_timewords = []
        self.decoded_image_containers = []

        self.housekeeping = []
        self.masks = []

        self.alloc_frames()

        self.extract_raw_images()
        self.decode_images()
        self.decompress_images()

    def get_buffer_sec(self, buffer_ts):
        return int(np.floor(buffer_ts))

    def get_buffer_ns(self, buffer_ts):
        return int( (buffer_ts - np.floor(buffer_ts))*1e9 )

    def alloc_frames(self):
        
        buffer_idx = 0

        while buffer_idx <  2048 - 5:
                flag = self.buffer[buffer_idx]

                if flag == DATA_FLAG:
                    buffer_idx = self.get_image_metadata_container(buffer_idx)
                elif flag == HK_FLAG:
                    self.process_housekeeping(buffer_idx)
                    buffer_idx += 53
                elif flag == MK_FLAG:
                    self.process_masks(buffer_idx)
                    buffer_idx += 23
                else:
                    buffer_idx += 1
    
    def get_image_metadata_container(self, buffer_idx):
        image_metadata_container = self.metadata_processor.process_metadata(buffer_idx, self.buffer)

        self.metadata_containers.append(image_metadata_container)

        return buffer_idx + image_metadata_container.frame_len


    def extract_raw_images(self):

        raw_image_containers = [None]*len(self.metadata_containers)
        timeword_containers = [None]*len(self.metadata_containers)

        for i, metadata in enumerate(self.metadata_containers):
            raw_image_containers[i] = self.raw_image_extractor.extract_raw_images(
                metadata, 
                self.buffer
            )

            timeword_containers[i] = self.raw_image_extractor.extract_image_timewords(
                metadata, 
                self.buffer
            )

        self.raw_image_containers = raw_image_containers
        self.timeword_containers = timeword_containers

    def decode_images(self):
        decoded_image_containers = [None]*len(self.raw_image_containers)

        for i, raw_image_container in enumerate(self.raw_image_containers):
            decoded_image_containers[i] = self.raw_image_decoder.decode_dual_channel_images(
                raw_image_container
            )

        self.decoded_image_containers = decoded_image_containers

    def decompress_images(self):
        decompressed_image_containers = [None]*len(self.decoded_image_containers)

        for i, decoded_image_container in enumerate(self.decoded_image_containers):
            decompressed_image_containers[i] = self.decoded_image_decompressor.decompress_image(
                decoded_image_container
            )

        self.decompressed_image_containers = decompressed_image_containers

    def process_housekeeping(self, buffer_idx):
        raw_housekeeping_packet = self.buffer[buffer_idx:buffer_idx + 53]

        self.housekeeping.append(
            housekeeping.process_housekeeping(raw_housekeeping_packet)
        )

    def process_masks(self, buffer_idx):
        pass

    def assemble_image_records(self): pass