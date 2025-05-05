from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    # name: str
    name :str = "Yogesh" # default value
    age : Optional[int] = None # Optional value
    # email : Optional[EmailStr] # Email validation
    cgpa : float = Field(gt=0, lt=10, default=5, description='represents student cgpa') # gt = greater than, lt = less than

# new_student = {'name':'Yogesh'} # if we pass sting it will work however if we pass int it will throw error
# new_student = {'name': 123} # this will throw error
# new_student = {}
# new_student = {'age': 35} # this will work
# new_student = {'age': '35'} # it will perform implicit type conversion

# new_student = {'email':'abc'} # it will throw error as it is not valid email
# new_student = {'email':'abc@gmail.com'} # this will work

new_student = {'cgpa': 9.5} # this will work, greater than 10 will through error

student = Student(**new_student)

student_dict = dict(student) # convert to dict

print(student)
print(student_dict)