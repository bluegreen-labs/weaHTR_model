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
    height=65,
    width=65,
    interpolation=1,
    p=1.0
  ),
  A.Rotate(
    limit=(-4, 4),
    interpolation=1,
    border_mode=3,
    p=0.5,
  ),
  # 20% tolerance on scaling
  # to introduce size variability
  A.RandomScale(
    scale_limit=0.2, 
    interpolation=1,
    p=0.8,
  ),
  A.GaussianBlur(
    blur_limit=(5, 13),
    p=0.5
  )
])

transform_sign = A.Compose([
  # always resize to a fixed 40px
  # first
  A.Resize(
    height=30,
    width=30,
    interpolation=1,
    p=1.0
  ),
  A.Rotate(
    limit=(-4, 4),
    interpolation=1,
    border_mode=3,
    p=0.5,
  ),
  # 20% tolerance on scaling
  # to introduce size variability
  A.RandomScale(
    scale_limit=0.2, 
    interpolation=1,
    p=0.8,
  ),
  A.GaussianBlur(
    blur_limit=(1, 11),
    p=0.5
  )
])

# random subsample of a background
# grid, including mild rotation
transform_grid = A.Compose([
  A.Rotate(
    limit=(-10, 10),
    interpolation=1,
    border_mode=3,
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
  A.GaussNoise(
    var_limit=(10.0, 50.0),
    mean=0,
    p=0.5
  ),
  A.RandomBrightnessContrast(
    brightness_limit=(-0.5, 0.5),
    contrast_limit=(-0.5, 0.5),
    p=0.5
  )
])
