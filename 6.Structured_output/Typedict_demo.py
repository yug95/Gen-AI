from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    email: str

new_person: Person = {'name':'Yogesh','age':35,'email':'xxx.com'}

print(new_person)