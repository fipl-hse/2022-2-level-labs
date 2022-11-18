"""
Programming 2022
Seminar 5
Introduction into classes, incupsulation
"""
class Student:

    def __init__(self, name: str, last_name: str, group_name: str, age: int):
        self._name = name
        self._last_name = last_name
        self._group_name = group_name
        self._age = age
        self._grades = {}

    def study(self) -> None:
        print(f'{self._name} is studying!')

    def sleep(self) -> None:
        print(f'{self._name} is sleeping!')

    def do_homework(self) -> None:
        print(f'{self._name} is doing homework!')

    def add_grade(self, subject: str, grade: int) -> None:
        if not isinstance(subject, str) or not isinstance(grade, int):
            print('INVALID VALUE')
            return
        if subject in self._grades:
            self._grades[subject].append(grade)
        else:
            self._grades[subject] = [grade]

    def __str__(self) -> str:
        return self._name + ' ' + self._last_name


student1 = Student('Andrej', 'K', '17FPL1', 22)
student1.add_grade('math', 7)

student2 = Student('Noah', 'B', '20FPL1', 19)
student2.add_grade('math', 10)


class StudentGroup:
    def __init__(self, group_name: str):
        self._group_name = group_name
        self._max_number_of_students = 15
        self._list_of_students = []

    def add_student(self, student: Student) -> None:
        if self.get_number_of_students() == self._max_number_of_students:
            print('There are too many students in the group')
            return
        self._list_of_students.append(student)

    def get_number_of_students(self) -> int:
        return len(self._list_of_students)

    def get_students(self) -> list[Student]:
        return self._list_of_students

student1 = Student('Andrej', 'K', '17FPL1', 22)
group1 = StudentGroup('20FPL3')
print(group1.get_number_of_students)
group1.add_student(student1)
print(group1.get_number_of_students())
print(group1.get_students())
