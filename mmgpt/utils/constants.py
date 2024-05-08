CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "log"

# Model Constants
IGNORE_INDEX = -100
# IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

DEFAULT_BOX_TOKEN = "<box>"
DEFAULT_BOX_START_TOKEN = "<box_start>"
DEFAULT_BOX_END_TOKEN = "<box_end>"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"

ROOT_PATH = '/path/to/dataset/'

CONVERSATION_DATA = dict(
    ################################### VL-Instruct Data ###############################
    blip_laion_cc_sbu_558k=dict(
        images='s3://vision-language-data/VisionPretrainDatasets/BLIP-CCSL-558k/',
        annotations='s3://vision-language-data/VisionPretrainDatasets/annotations/blip_laion_cc_sbu_558k.json',
        frequency=1,
    ),
    llava665k=dict(
        images='',
        annotations='/path/to/dataset/LLaVA1.5/llava_v1_5_mix665k_s3path.json',
        frequency=1,
    ),
    llava665k_refine=dict(
        images='',
        annotations='/path/to/dataset/LLaVA1.5/llava_v1_5_665k_refine_s3path.json',
        frequency=1,
    ),
    track_sft=dict(
        images='',
        annotations='/path/to/datasets/vl/annotations/mix_sft_track_30k.json',
        frequency=1,
    ),
    track_sft_v1=dict(
        images='',
        annotations='/path/to/datasets/vl/annotations/new_format/merlin_sft_70k-valid.json',
        frequency=1,
    ),
    track_sft_v2=dict(
        images='',
        annotations='/data/hypertext/danielyu/datasets/vl/annotations/new_format/merlin_sft_70k_v2-valid.json',
        frequency=1,
    ),
    track_sft_v3=dict(
        images='',
        annotations='/data/hypertext/danielyu/datasets/vl/annotations/new_format/merlin_sft_90k_v3.json',
        frequency=1,
    ),
    track_sft_pretrain_44k=dict(
        images='',
        annotations='/data/hypertext/danielyu/datasets/vl/annotations/new_format/mix_sample_pretrain_44k.json',
        frequency=1,
    ),
    track_sft_46k_v3=dict(
        images='',
        annotations='/data/hypertext/danielyu/datasets/vl/annotations/new_format/mix_sample_track_sft_46k_v3.json',
        frequency=1,
    ),


INTERLEAVE_DATA = dict(
    ################################### Caption-related Data ####################################
    cococaption=dict(
        images='s3://vision-language-data/VisionPretrainDatasets/COCO/',
        annotations='s3://vision-language-data/VisionPretrainDatasets/annotations/cococap_interleave_38k.json',
        use_eos_for_each_turn=True,
        frequency=1,
    ),
    nocaps=dict(
        images='s3://vision-language-data/VisionPretrainDatasets/NoCaps-4.5k/',
        annotations='s3://vision-language-data/VisionPretrainDatasets/annotations/nocaps_interleave_1.5k.json',
        use_eos_for_each_turn=True,
        frequency=1,
    ),
    flickr30k=dict(
        images='s3://vision-language-data/VisionPretrainDatasets/Flickr-30k/',
        annotations='s3://vision-language-data/VisionPretrainDatasets/annotations/flickr30k_interleave_10k.json',
        use_eos_for_each_turn=True,
        frequency=1,
    ),
    flickr30k_cn=dict(
        images='s3://vision-language-data/VisionPretrainDatasets/Flickr-30k/',
        annotations='s3://vision-language-data/VisionPretrainDatasets/annotations/flickr30k_cn_interleave_10k.json',
        use_eos_for_each_turn=True,
        frequency=1,
    ),
    textcaps=dict(
        images='s3://vision-language-data/VisionPretrainDatasets/TextCaps-25k/',
        annotations='s3://vision-language-data/VisionPretrainDatasets/annotations/textcaps_interleave_7k.json',
        use_eos_for_each_turn=True,
        frequency=1,
    ),
    ################################### Detection-related Data ####################################
    obj365=dict(
        images='s3://vision-language-data/VisionPretrainDatasets/Objects365/',
        annotations='/path/to/datasets/vl/annotations/objv2_train_interleave_det345k_correct.json',
        use_eos_for_each_turn=True,
        frequency=1,
    ),
    obj365_cn=dict(
        images='s3://vision-language-data/VisionPretrainDatasets/Objects365/',
        annotations='/path/to/datasets/vl/annotations/objv2_train_interleave_det345k_cn_correct.json',
        use_eos_for_each_turn=True,
        frequency=1,
    ),
    openimages=dict(
        images='s3://vision-language-data/VisionPretrainDatasets/OpenImages/train/',
        annotations='/path/to/datasets/vl/annotations/openimages_interleave_det348k.json',
        use_eos_for_each_turn=True,
        frequency=1,
    ),
    openimages_cn=dict(
        images='s3://vision-language-data/VisionPretrainDatasets/OpenImages/train/',
        annotations='/path/to/datasets/vl/annotations/openimages_interleave_det348k_cn.json',
        use_eos_for_each_turn=True,
        frequency=1,
    ),
    lvis=dict(
        images='s3://data-transfer-tos-shanghai-818/vision-language-data/VisionPretrainDatasets/COCO/',
        annotations='/path/to/datasets/vl/annotations/lvis_train_interleave_det20k.json',
        use_eos_for_each_turn=True,
        frequency=1,
    ),
    lvis_cn=dict(
        images='s3://data-transfer-tos-shanghai-818/vision-language-data/VisionPretrainDatasets/COCO/',
        annotations='/path/to/datasets/vl/annotations/lvis_train_interleave_det20k_cn.json',
        use_eos_for_each_turn=True,
        frequency=1,
    ),
)


def get_part(num):
    num = num // 10000
    return num if num < 3 else 3

PAIR_WEBDATA = dict(
    ############################ Laion 2B series ##########################
    laion2b_10m=dict(
        path=[f"s3://laion2b-en/server1-Cube-II-21A834208.nori/{i:08d}.tar" for i in range(5685)],
        size=10000000,
        merge_round=12
    ),
    laion2b_10m_6merge=dict(
        path=[f"s3://laion2b-en/server1-Cube-II-21A834208.nori/{i:08d}.tar" for i in range(5685)],
        size=10000000,
        merge_round=6
    ),
    laion2b_20m_6merge=dict(
        path=[f"s3://laion2b-en/server1-Cube-II-21A834208.nori/{i:08d}.tar" for i in range(5685)],
        size=20000000,
        merge_round=6
    ),
    laion2b_5m=dict(
        path=[f"s3://laion2b-en/server1-Cube-II-21A834208.nori/{i:08d}.tar" for i in range(5685)],
        size=5000000,
        merge_round=12
    ),
    ############################ TAISU 100M series ##########################
    taisu_100m=dict(
        path=f"s3://vision-language-data/taisu-tarfiles/*/*.tar",
        size=100000000,
        merge_round=12
    ),
    taisu_20m_6merge=dict(
        path=f"s3://vision-language-data/taisu-tarfiles/*/*.tar",
        size=20000000,
        merge_round=6
    ),
    ############################ Laion 400M series ##########################
    laion400m_100m=dict(
        path=[f"s3://vision-language-data/Laion400M/laion400m_part{get_part(i)}/{i:05d}.tar" for i in range(15000)],
        size=100000000,
        merge_round=12
    ),
    laion400m_70m=dict(
        path=[f"s3://vision-language-data/Laion400M/laion400m_part{get_part(i)}/{i:05d}.tar" for i in range(15000)],
        size=70000000,
        merge_round=12
    ),
    laion400m_50m=dict(
        path=[f"s3://vision-language-data/Laion400M/laion400m_part{get_part(i)}/{i:05d}.tar" for i in range(7500)],
        size=50000000,
        merge_round=12
    ),
    laion400m_10m=dict(
        path=[f"s3://vision-language-data/Laion400M/laion400m_part{get_part(i)}/{i:05d}.tar" for i in range(7500)],
        size=10000000,
        merge_round=12
    ),
    laion400m_5m=dict(
        path=[f"s3://vision-language-data/Laion400M/laion400m_part{get_part(i)}/{i:05d}.tar" for i in range(7500)],
        size=5000000,
        merge_round=12
    ),
    ############################ Laion chinese 100M series ##########################
    laion_cn_100m=dict(
        path=f"s3://vision-language-data/laion-chinese-100m-tarfiles/*/*.tar",
        size=100000000,
        merge_round=12
    ),
    laion_cn_70m=dict(
        path=f"s3://vision-language-data/laion-chinese-100m-tarfiles/*/*.tar",
        size=70000000,
        merge_round=12
    ),
    laion_cn_50m=dict(
        path=f"s3://vision-language-data/laion-chinese-100m-tarfiles/*/*.tar",
        size=50000000,
        merge_round=12
    ),
    laion_cn_10m=dict(
        path=f"s3://vision-language-data/laion-chinese-100m-tarfiles/*/*.tar",
        size=10000000,
        merge_round=12
    ),
    laion_cn_10m_6merge=dict(
        path=f"s3://vision-language-data/laion-chinese-100m-tarfiles/*/*.tar",
        size=10000000,
        merge_round=6
    ),
    laion_cn_5m=dict(
        path=f"s3://vision-language-data/laion-chinese-100m-tarfiles/*/*.tar",
        size=5000000,
        merge_round=12
    ),
    ############################ Synthdog series ##########################
    synthdog_224_10m=dict(
        path=f"s3://vision-language-data/synthdog-224-tarfiles/*/*.tar",
        size=10000000,
        merge_round=12
    ),
    synthdog_10m=dict(
        path=f"s3://vision-language-data/synthdog-tarfiles/*/*.tar",
        size=10000000,
        merge_round=12
    ),
    synthdog_2m_6merge=dict(
        path=f"s3://vision-language-data/synthdog-tarfiles/*/*.tar",
        size=2000000,
        merge_round=6
    ),

    ############################ Detection series ##########################
    grit_5m=dict(
        path=f"s3://vision-language-data/grit-5m-tarfiles/*.tar",
        size=5000000,
        merge_round=12
    ),
    grit_5m_6merge=dict(
        path=f"s3://vision-language-data/grit-5m-tarfiles/*.tar",
        size=5000000,
        merge_round=6
    ),
    grit_2_5m=dict(
        path=f"s3://vision-language-data/grit-5m-tarfiles/*.tar",
        size=2500000,
        merge_round=12
    ),
    det_224_5m=dict(
        path=f"s3://vision-language-data/detection-224-tarfiles/*/*.tar",
        size=5000000,
        merge_round=8
    ),
    det_3m_4merge=dict(
        path=f"s3://vision-language-data/detection-tarfiles/*/*.tar",
        size=3000000,
        merge_round=4
    ),
    det_224_3m_en=dict(
        path=f"s3://vision-language-data/detection-224-tarfiles/*-en/*.tar",
        size=3000000,
        merge_round=8
    ),
    det_3m_en=dict(
        path=f"s3://vision-language-data/detection-tarfiles/*-en/*.tar",
        size=3000000,
        merge_round=8
    ),
    det_3m_en_4merge=dict(
        path=f"s3://vision-language-data/detection-tarfiles/*-en/*.tar",
        size=3000000,
        merge_round=4
    ),
    det_5m_v1_en_4merge=dict(
        path=f"/data/hypertext/data/data/dataset/det-tarfiles-v1/*.tar",
        size=5000000,
        merge_round=4
    ),
    det_1_5m_en=dict(
        path=f"s3://vision-language-data/detection-tarfiles/*-en/*.tar",
        size=1500000,
        merge_round=8
    ),
    track_224_2m=dict(
        path=f"/data/hypertext/data/data/dataset/track-224-tarfiles/*.tar",
        size=2000000,
        merge_round=5
        # merge_round=4
    ),
    track_224_1m=dict(
        path=f"/data/hypertext/data/data/dataset/track-224-tarfiles/*.tar",
        size=1000000,
        merge_round=5
        # merge_round=4
    ),
    track_2m=dict(
        path=f"/data/hypertext/data/data/dataset/track-tarfiles/*.tar",
        size=2000000,
        merge_round=5
        # merge_round=4
    ),
    track_1m=dict(
        path=f"/data/hypertext/data/data/dataset/track-tarfiles/*.tar",
        size=1000000,
        merge_round=5
        # merge_round=4
    ),
    track_1m_2merge=dict(
        path=f"/data/hypertext/data/data/dataset/track-tarfiles/*.tar",
        size=1000000,
        merge_round=2
    ),
    track_1m_v1_2merge=dict(
        path=f"/data/hypertext/data/data/dataset/track-tarfiles-v1/*.tar",
        size=1000000,
        merge_round=2
    ),
    track_1m_v2_2merge=dict(
        path=f"/data/hypertext/data/data/dataset/track-tarfiles-v2/*.tar",
        size=1000000,
        merge_round=2
    ),
)

INTERLEAVE_WEBDATA = dict(
    oblisc_1m=dict(
        path=f"s3://vision-language-data/oblisc-tarfiles/0/*.tar",
        size=1000000,
    ),
)