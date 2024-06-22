import random
import albumentations as A

def generate_random_digits(
  min_length=1,
  max_length=3,
  include_decimal=True,
  include_minus=False
  ):
  """
  Generates a random string of digits with or without one trailing decimal.

  Args:
      min_length (int, optional): Minimum length of the string. Defaults to 1.
      max_length (int, optional): Maximum length of the string before the decimal. Defaults to 5.
      include_decimal (bool, optional): Whether to include a trailing decimal. Defaults to True.
      include_minus (bool, optional): Allow negative values. Defaults to False.

  Returns:
      list: A random list of digits and decimal points.
      str: The collapsed list as a a decimal value.
  """

  # Generate a random integer length between min_length and max_length
  length = random.randint(min_length, max_length)
  
  # Generate a random string of digits
  digits = [random.randint(0, 9) for _ in range(length)]
  digits = [str(x) for x in digits]
  value = ''.join(digits)

  # Add a trailing decimal with a random probability
  if include_decimal and random.random() < 0.5:
    decimal_separator = random.choice([".", ","])
    digits.append(decimal_separator)
    decimal_digit = str(random.randint(0, 9))
    digits.append(decimal_digit)
    value += decimal_separator + decimal_digit
    
  if include_minus and random.random() < 0.5:
    digits.insert(0, "-")
    value = "-" + value
  
  return digits, value

transform_number = A.Compose([
  # always resize to a fixed 40px
  # first
  A.Resize(
    height=40,
    width=40,
    interpolation=1,
    always_apply=None,
    p=1.0
  ),
  A.Rotate(
    limit=(-10, 10),
    interpolation=1,
    border_mode=4,
    value=None,
    mask_value=None,
    rotate_method="largest_box",
    crop_border=False, # check outcome on this
    always_apply=None,
    p=0.5,
  ),
  # 10% tolerance on scaling
  # to introduce size variability
  A.RandomScale(
    scale_limit=0.2, 
    interpolation=1,
    always_apply=None,
    p=0.5,
  ),
  A.Sharpen(
    alpha=(0.2, 0.5),
    lightness=(0.5, 1.0),
    always_apply=None,
    p=0.5
  ),
  A.GaussianBlur(
    blur_limit=(3, 3),
    sigma_limit=0,
    always_apply=None,
    p=0.5
  )
])

# random subsample of a background
# grid, including mild rotation
transform_grid = A.Compose([
  A.Rotate(
    limit=(-10, 10),
    interpolation=1,
    border_mode=4,
    value=None,
    mask_value=None,
    rotate_method="largest_box",
    crop_border=False,
    always_apply=None,
    p=0.2,
  ),
  A.RandomCrop(
        height=200,
        width=200,
        always_apply=None,
        p=1.0
   )
])

transform_image = A.Compose([
    A.ElasticTransform(
    alpha=1,
    sigma=50,
    alpha_affine=50,
    interpolation=1,
    border_mode=4,
    value=None,
    mask_value=None,
    always_apply=None,
    approximate=False,
    same_dxdy=False,
    p=0.5,
  ),
  A.GaussNoise(
    var_limit=(10.0, 50.0),
    mean=0,
    per_channel=True,
    always_apply=None,
    p=0.5
  ),
  A.RandomBrightnessContrast(
    brightness_limit=(-0.2, 0.2),
    contrast_limit=(-0.2, 0.2),
    brightness_by_max=True,
    always_apply=None,
    p=0.5
  )
])
