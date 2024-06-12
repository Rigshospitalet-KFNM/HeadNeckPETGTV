#region Imports
# Python standard library
from datetime import datetime, timedelta
from logging import DEBUG
from subprocess import run as run_subprocess, CompletedProcess
from pathlib import Path
from os import environ, getcwd

# Third party Packages
import nibabel
import numpy
from pydicom import dcmread
from rt_utils import RTStructBuilder
from dicomnode.dicom.dimse import Address
from dicomnode.server.grinders import ListGrinder
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
  nii_ct_path_destination = 'HNC04_000_CT.nii.gz'
  cropped_img.to_filename(nii_ct_path_destination)

  return nii_ct_path_destination


timestamp_format = "%Y%m%d%H%M%S.%f"

def dose_calculation(initial_dose,
                     halflife_seconds,
                     decay_time_delta: timedelta ,
                     ):
  return initial_dose * numpy.exp(numpy.log(2) / halflife_seconds * (-decay_time_delta.seconds))


def suv_rescale(image: numpy.ndarray, dose:float, patient_weight: float):
  return image / (dose / patient_weight)

output_address = Address(
  '172.16.167.172',
  11112,
  'STORESCP',
)


#region Inputs
class PET_Input(AbstractInput):
  required_values = {
    0x00080060 : 'PT'
  }

  def validate(self) -> bool:
    return self.images > 0

  image_grinder = ListGrinder()

class CT_Input(AbstractInput):
  required_values = {
    0x00080060 : 'CT'
  }

  def validate(self) -> bool:
    return self.images > 0

  image_grinder = ListGrinder()

#region Pipeline
class PET_GTV_Pipeline(AbstractQueuedPipeline):
  input = {
    'PET' : PET_Input,
    'CT'  : CT_Input,
  }

  study_expiration_days=1
  ip='0.0.0.0'
  port=11112
  log_output = Path(LOG_PATH)
  ae_title = "PETGTVAISEG"
  data_directory = ARCHIVE_PATH
  processing_directory = WORKING_PATH
  log_level = DEBUG

  def log_subprocess(self, output: CompletedProcess, process_name: str):
    if output.returncode != 0:
      self.logger.error(f"{process_name} return code: {output.returncode}")
      self.logger.error(f"{process_name} stdout: {output.stdout.decode()}")
      self.logger.error(f"{process_name} stderr: {output.stderr.decode()}")

  def process(self, input_data: InputContainer) -> PipelineOutput:
    ct_path = input_data.paths['CT']
    pet_path = input_data.paths['PET']
    pivot_pet_dataset = input_data['PET'][0]

    # region SUV calculation
    patient_weight = pivot_pet_dataset.PatientWeight
    acquisition_date_str = pivot_pet_dataset.AcquisitionDate
    acquisition_time_str = pivot_pet_dataset.AcquisitionTime
    acquisition_datetime = datetime.strptime(f"{acquisition_date_str}{acquisition_time_str}",timestamp_format)
    tracer_info = pivot_pet_dataset.RadiopharmaceuticalInformationSequence[0]
    injection_datetime = datetime.strptime(tracer_info.RadiopharmaceuticalStartDateTime, timestamp_format)
    decay_delta_time = acquisition_datetime - injection_datetime
    injection_dose_MBq = tracer_info.RadionuclideTotalDose / 1_000_000
    halflife_seconds = tracer_info.RadionuclideHalfLife
    corrected_dose = dose_calculation(injection_dose_MBq, halflife_seconds, decay_delta_time)


    # Dicom to nifti conversion
    cwd = Path(getcwd())
    pet_destination_path = "HNC04_000_PET.nii.gz"
    ct_command = [DCM2NIIX, '-o', str(cwd), '-f', 'ct',str(ct_path)]
    self.log_subprocess(run_subprocess(ct_command, capture_output=True),
                        "dcm2niix ct")


    pet_command = [DCM2NIIX, '-o', str(cwd), '-f', 'pet', str(pet_path)]
    self.log_subprocess(run_subprocess(pet_command, capture_output=True),
                        "dcm2niix pet")

    ct_nifti_path = crop_to_350_mm('ct.nii')

    #region Resampling
    resample_command = [
      'reg_resample',
      '-ref', ct_nifti_path,
      '-flo', 'pet.nii',
      '-res', pet_destination_path,
    ]

    self.log_subprocess(run_subprocess(resample_command, capture_output=True),
                        'Pet Resample')


    pet_image = nibabel.load(pet_destination_path)
    pet_data = pet_image.get_fdata()
    pet_data = suv_rescale(pet_data, corrected_dose, patient_weight)
    pet_image = nibabel.Nifti1Image(pet_data, pet_image.affine, pet_image.header)
    nibabel.save(pet_image, pet_destination_path)

    #
    segmentation_path = cwd / "segmentation.nii.gz"
    podman_command = ['podman',
                    'run',
                    '--security-opt=label=disable',
                    '--device=nvidia.com/gpu=all',
                    '-v',
                    f'{str(cwd)}:/usr/src/app/dataset',
                    'depict/hnc_pet_gtv:latest',
                    pet_destination_path,
                    ct_nifti_path,
                    "segmentation.nii.gz"
                  ]
    self.log_subprocess(run_subprocess(podman_command, capture_output=True),
                        'Podman')

    segmentation: nibabel.nifti1.Nifti1Image = nibabel.load(str(segmentation_path))
    pipeline_mask = segmentation.get_fdata().astype(numpy.bool_)

    # Resize mask such that fits with the CT
    empty_mask = numpy.zeros((len(input_data.datasets['CT']) - pet_data.shape[0],
                              pet_data.shape[1],
                              pet_data.shape[2]),
                              dtype=numpy.bool_)

    mask = numpy.concatenate((empty_mask, pipeline_mask), axis=0)
    self.logger.error(f"Pipeline Mask shape: {pipeline_mask.shape}")
    self.logger.error(f"Mask shape: {mask.shape}")
    self.logger.error(f"CT images: {len(input_data.datasets['CT'])}")


    rt_struct = RTStructBuilder.create_new(
      str(ct_path)
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

  def post_init(self) -> None:
    cwd = getcwd()
    self.logger.info(f"Started to run the process at {cwd}")

#region __main__
if __name__ == '__main__':
  pipeline = PET_GTV_Pipeline()
  pipeline.open()
