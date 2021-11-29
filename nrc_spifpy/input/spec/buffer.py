import numpy as np

from nrc_spifpy.input.spec.image import ImageMetadataProcessor
from nrc_spifpy.input.spec.image import RawImageExtractor
from nrc_spifpy.input.spec.image import RawImageDecoder
from nrc_spifpy.input.spec.image import DecodedImageDecompressor
from nrc_spifpy.input.spec.image import ImageRecordAssembler
from nrc_spifpy.input.spec.image import AssembledImageRecordContainer

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

        self.assembled_images = {
            'h':None,
            'v':None
        }

        self.housekeeping = np.array([], dtype = housekeeping.housekeeping_template)
        self.masks = []

        self.alloc_frames()

        self.extract_raw_images()
        self.decode_images()
        self.decompress_images()

        self.assemble_image_records()

    def get_buffer_sec(self, buffer_ts):
        return int(np.floor(buffer_ts))

    def get_buffer_ns(self, buffer_ts):
        return int( (buffer_ts - np.floor(buffer_ts))*1e9 )

    def alloc_frames(self):
        
        buffer_index = 0

        while buffer_index <  2048 - 5:
                flag = self.buffer[buffer_index]

                if flag == DATA_FLAG:
                    buffer_index = self.get_image_metadata_container(buffer_index)
                elif flag == HK_FLAG:
                    self.process_housekeeping(buffer_index)
                    buffer_index += 53
                elif flag == MK_FLAG:
                    self.process_masks(buffer_index)
                    buffer_index += 23
                else:
                    buffer_index += 1
    
    def get_image_metadata_container(self, buffer_index):
        image_metadata_container = self.metadata_processor.process_metadata(buffer_index, self.buffer)

        self.metadata_containers.append(image_metadata_container)

        return buffer_index + image_metadata_container.frame_len


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

    def process_housekeeping(self, buffer_index):
        raw_housekeeping_packet = self.buffer[buffer_index:buffer_index + 53]

        self.housekeeping = np.append(
            self.housekeeping,
            housekeeping.process_housekeeping(
                self.buffer_id, 
                self.buffer_sec, 
                self.buffer_ns,
                buffer_index,
                raw_housekeeping_packet
            )
        )

    def process_masks(self, buffer_index):
        pass

    def assemble_image_records(self):

        assembled_images_h, assembled_images_v = self.image_record_assembler.assemble_images(
            self.buffer_id,
            self.buffer_sec,
            self.buffer_ns,
            self.metadata_containers,
            self.timeword_containers,
            self.decompressed_image_containers,
            self.housekeeping
        )

        self.assembled_images['h'] = assembled_images_h
        self.assembled_images['v'] = assembled_images_v
    

