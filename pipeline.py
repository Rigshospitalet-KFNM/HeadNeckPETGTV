#region Imports
# Python standard library
from subprocess import run as run_subprocess
from pathlib import Path
from os import environ, getcwd

# Third party Packages
import nibabel
import numpy
from rt_utils import RTStructBuilder
from dicomnode.dicom.dimse import Address
from dicomnode.server.grinders import IdentityGrinder
from dicomnode.server.nodes import AbstractQueuedPipeline
from dicomnode.server.input import AbstractInput
from dicomnode.server.output import DicomOutput, PipelineOutput, FileOutput, MultiOutput
from dicomnode.server.pipeline_tree import InputContainer


#region Environment Setup
ENVIRONMENT_ARCHIVE_PATH = "PIPELINE_ARCHIVE_PATH"
ENVIRONMENT_ARCHIVE_PATH_VALUE = environ.get(ENVIRONMENT_ARCHIVE_PATH,
                                             "/tmp/pet_gtx_pipeline_archive")

ARCHIVE_PATH = Path(ENVIRONMENT_ARCHIVE_PATH_VALUE)
if not ARCHIVE_PATH.exists():
  ARCHIVE_PATH.mkdir()

ENVIRONMENT_WORKING_PATH = "PIPELINE_WORKING_PATH"
ENVIRONMENT_WORKING_PATH_VALUE = environ.get(ENVIRONMENT_WORKING_PATH,
                                             "/tmp/pet_gtx_pipeline_working")

WORKING_PATH = Path(ENVIRONMENT_WORKING_PATH_VALUE)
if not WORKING_PATH.exists():
  WORKING_PATH.mkdir()


ENVIRONMENT_LOG_PATH = "PIPELINE_LOG_PATH"
ENVIRONMENT_LOG_PATH_VALUE = environ.get(ENVIRONMENT_LOG_PATH,
                                         "/var/log/pipeline")

LOG_PATH = Path(ENVIRONMENT_LOG_PATH_VALUE)


ENVIRONMENT_DCM2NIIX_PATH = "PIPELINE_DCM2NIIX"
DCM2NIIX = environ.get(ENVIRONMENT_DCM2NIIX_PATH,
                                             "dcm2niix")

which_output = run_subprocess(['which', DCM2NIIX], capture_output=True)
if(not len(which_output.stdout)):
  raise Exception("COULD NOT FIND DCM2NIIX")


ENVIRONMENT_RESAMPLE_PATH = "PIPELINE_RESAMPLE"
RESAMPLE = environ.get(ENVIRONMENT_RESAMPLE_PATH,
                                             "reg_resample")

which_output = run_subprocess(['which', RESAMPLE], capture_output=True)
if(not len(which_output.stdout)):
  raise Exception("COULD NOT FIND RESAMPLE program")


#region Setup
def crop_to_350_mm(nii_ct_path : Path):
  # get n_slices in first 35 cm
  img = nibabel.load(nii_ct_path)

  slice_thickness = img.header['pixdim'][3]
  tot_slices = img.header['dim'][3]
  n_slices = int(numpy.ceil(350 / slice_thickness))
  cropped_img = img.slicer[:,:,tot_slices-n_slices:tot_slices]
  nii_ct_path_destination = str(nii_ct_path) + "_cropped"
  cropped_img.to_filename(nii_ct_path_destination)

  return nii_ct_path_destination

output_address = Address(
  '10.19.144.12',
  104,
  'LILJEFORS',
)


#region Inputs
class PET_Input(AbstractInput):
  required_values = {
    0x00080060 : 'PT'
  }

  def validate(self) -> bool:
    return self.images > 0

  image_grinder = IdentityGrinder()

class CT_Input(AbstractInput):
  required_values = {
    0x00080060 : 'CT'
  }

  def validate(self) -> bool:
    return self.images > 0

  image_grinder = IdentityGrinder()

#region Pipeline
class PET_GTV_Pipeline(AbstractQueuedPipeline):
  input = {
    'PET' : PET_Input,
    'CT'  : CT_Input,
  }


  port=11113
  log_output = Path(LOG_PATH)
  ae_title = "PETGTVAISEG"
  data_directory = ARCHIVE_PATH
  processing_directory = WORKING_PATH

  def process(self, input_data: InputContainer) -> PipelineOutput:
    ct_path = input_data.paths['CT']
    pet_path = input_data.paths['PET']

    cwd = Path(getcwd())
    ct_nifti = cwd / "ct.nii.gz"
    pet_nifti = cwd / "pet.nii.gz"

    run_subprocess([DCM2NIIX, '-o', str(ct_nifti), str(ct_path)])
    run_subprocess([DCM2NIIX, '-o', str(pet_nifti), str(pet_path)])

    ct_nifti_cropped = crop_to_350_mm(ct_nifti)

    segmentation_path = cwd / "seg.nii.gz"

    run_subprocess(['podman',
                    'run',
                    '--security-opt=label=disable',
                    '--device=nvidia.com/gpu=all',
                    '-v',
                    f'{str(cwd)}:/usr/src/app/dataset',
                    'depict/hnc_pet_gtv:latest',
                    str(pet_nifti),
                    str(ct_nifti_cropped),
                    str(segmentation_path)
                  ])

    segmentation: nibabel.nifti1.Nifti1Image = nibabel.load(str(segmentation_path))
    mask = segmentation.get_fdata()

    # Assume this works!
    rt_struct = RTStructBuilder.create_new(
      str(pet_path)
    )

    rt_struct.add_roi(
      mask=mask,
      color=[[255,255,255]],
      description="PET GTV AI Segmentation"
    )

    rt_dataset = rt_struct.ds
    # The output dataset to change
    rt_dataset.SeriesDescription = "PET GTV AT Segmentation"

    return DicomOutput([
      [(output_address, rt_dataset)], self.ae_title
    ])

#region __main__
if __name__ == '__main__':
  pipeline = PET_GTV_Pipeline()
  pipeline.open()
