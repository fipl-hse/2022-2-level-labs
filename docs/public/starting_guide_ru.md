# Подготовка к прохождению курса

Перед началом прохождения курса "Программирование для лингвистов" каждому студенту необходимо сделать несколько шагов,
которые подготовят необходимые инструменты к дальнейшей работе:

1. [Установить интерпретатор языка программирования Python](#installing-python)
2. [Установить систему контроля версий Git](#installing-git)
3. [Установить среду разработки PyCharm](#installing-pycharm)
4. [Зарегистрироваться на платформе GitHub](#github-registration)
5. [Создать форк репозитория](#creating-fork)
6. [Клонировать форк репозитория для локальной работы](#cloning-fork)
7. [Создать проект в среде разработки PyCharm](#creating-project-in-pycharm)
8. [Изменить исходный код и отправить изменения в удалённый форк](#working-pipeline)
9. [Создать Pull Request](#creating-pull-request)
10. [Продолжить работу](#continue-working)

## <a name="installing-python"></a>Установка интерпретатора языка программирования Python

Чтобы установить интерпретатор языка программирования Python на свой компьютер, выполните следующие шаги:
1. Скачайте установочный файл для своей системы с [официального сайта](https://www.python.org/downloads/)
    
    ![](../images/starting_guide/download_python.png)
    * **NB**: Для скачивания нажмите кнопку `Download Python 3.XX.XX`
    * **NB №2**: Версия Python должна быть >= 3.10! 
2. Запустите установочный файл и следуйте указаниям по установке
3. Проверьте корректность установки:
   1. Откройте терминал и выполните следующую команду:
      * Mac: `python3 -V`
      * Windows: `python -V`
   2. Вы должны увидеть строку, похожую на следующую: `Python 3.10.2`

* Как открыть терминал:
  * [Инструкция для Windows](https://docs.microsoft.com/ru-ru/powershell/scripting/windows-powershell/starting-windows-powershell?view=powershell-7.2)
  * [Инструкция для MacOS](https://support.apple.com/ru-ru/guide/terminal/apd5265185d-f365-44cb-8b09-71a064a42125/mac)

## <a name="installing-git"></a>Установка системы контроля версий Git

Чтобы установить систему контроля версий Git, выполните следующие шаги:
1. Скачайте установочный файл для своей системы с [официального сайта](https://git-scm.com)
    
    ![](../images/starting_guide/git_main_page.png)
    * **NB**: Для скачивания нажмите кнопку `Download for <название ОС>`
2. Запустите установочный файл и следуйте указанием по установке
3. Проверьте корректность установки:
   1. Откройте терминал и выполните следующую команду:
      * `git`
   2. Вы должны увидеть строку, похожую на следующую: `usage: git ...`

* Как открыть терминал:
  * [Инструкция для Windows](https://docs.microsoft.com/ru-ru/powershell/scripting/windows-powershell/starting-windows-powershell?view=powershell-7.2)
  * [Инструкция для MacOS](https://support.apple.com/ru-ru/guide/terminal/apd5265185d-f365-44cb-8b09-71a064a42125/mac)

## <a id="installing-pycharm"></a>Установка среды разработки PyCharm

Чтобы установить среду разработки PyCharm, выполните следующие шаги:
1. Скачайте установочный файл для своей системы с [официального сайта](https://www.jetbrains.com/pycharm/download/)

    ![](../images/starting_guide/pycharm_download.png)
    * **NB**: Для скачивания нажмите кнопку `Download`
    * **NB №2**: Рекомендуется использовать Community Edition, так как эта версия бесплатна
    * **NB №3**: Так как Вы являетесь студентом ВШЭ, Вы можете использовать и Professional Edition, зарегистрировавшись с учебной почтой. 
      * Данная версия предоставляет [дополнительные возможности](https://www.jetbrains.com/ru-ru/products/compare/?product=pycharm&product=pycharm-ce).
2. Запустите установочный файл и следуйте указанием по установке
3. Проверьте корректность установки:
   1. Откройте PyCharm из меню приложений
   2. Вы должны увидеть похожий интерфейс:
        
       ![](../images/starting_guide/pycharm_project_selection.png)

## <a name="github-registration"></a>Регистрация на платформе GitHub

Чтобы зарегистрироваться на платформе GitHub, выполните следующие шаги:
1. Откройте [главную страницу платформы](https://github.com)
2. В верхнем правом углу нажмите кнопку `Sign up`:

    ![](../images/starting_guide/github_sign_up.png)
3. Пройдите регистрацию
   * **NB**: Рекомендуется использовать личную почту, чтобы после окончания учёбы не пришлось менять почту с учебной на личную
   * **NB №2**: Рекомендуется (но не обязательно) в качестве логина использовать фамилию и имя
     * Пример: AndreiKashchikhin

## <a name="creating-fork"></a>Создание форка репозитория

Чтобы создать форк репозитория на платформе GitHub, выполните следующие шаги:
1. Откройте сайт репозитория, который Вам прислал преподаватель
2. В верхнем правом углу нажмите кнопку `Fork`:

    ![](../images/starting_guide/github_forking.png)
3. На открывшейся странице нажмите кнопку `Create Fork`

    ![](../images/starting_guide/github_forking_2.png)
4. Форк создан. Обратите внимание на ссылку в адресной строке браузера: она будет содержать **имя Вашего GitHub пользователя** и название репозитория:
   * https://github.com/<имя-Вашего-пользователя>/202X-2-level-labs
    ![](../images/starting_guide/github_forking_3.png)

## <a name="cloning-fork"></a>Клонирование форка репозитория для локальной работы

Чтобы склонировать форк на Вашу систему, выполните следующие шаги:
1. Откройте сайт Вашего форка, который Вы создали на предыдущем шаге
2. Нажмите кнопку `Code`, выберите `HTTPS` и нажмите кнопку копирования:

    ![](../images/starting_guide/cloning_repository.png)
3. Откройте терминал и перейдите в удобную папку:
   * Чтобы переходить из папки в папку в терминале, используйте команду `cd <название-папки>`
     * Пример: `cd work`
4. Выполните следующую команду для клонирования репозитория:
   * `git clone <ссылка-на-ваш-форк>`
     * Пример: `git clone https://github.com/WhiteJaeger/2022-2-level-labs`
   * Ссылку на форк Вы скопировали ранее на шаге №2 

* Как открыть терминал:
  * [Инструкция для Windows](https://docs.microsoft.com/ru-ru/powershell/scripting/windows-powershell/starting-windows-powershell?view=powershell-7.2)
  * [Инструкция для MacOS](https://support.apple.com/ru-ru/guide/terminal/apd5265185d-f365-44cb-8b09-71a064a42125/mac)

## <a name="creating-project-in-pycharm"></a>Создание проекта в среде разработки PyCharm

Чтобы создать проект и работать с Вашим форком в среде разработки PyCharm, выполните следующие шаги:

1. Откройте PyCharm и нажмите кнопку `Open`

   ![](../images/starting_guide/opening_project.png) 
2. В открывшемся окне выберите папку с форком, который Вы склонировали на шаге "Клонирование форка репозитория для локальной работы"
3. В открывшемся окне нажмите кнопку `OK`

    ![](../images/starting_guide/setting_interpreter.png)
    * **NB**: Если в поле `Base Interpreter` версия Python < 3.9, то нажмите на `Python 3.X` и из выпадающего списка выберите более новую версию
4. Проект создан, слева Вы можете увидеть файлы проекта

    ![](../images/starting_guide/initial_project_setup.png)

## <a name="working-pipeline"></a>Изменение исходного кода и отправка изменений в удалённый форк

Основную работу Вы будете вести в файле `main.py` в папке с каждой лабораторной работой.

Процесс выглядит следующим образом:
1. Вы изменяете исходный код в файле `main.py`
2. Вы фиксируете изменения с помощью системы контроля версий `git`
3. Вы отправляете изменения в удалённый форк

Далее будет пример этого процесса.

### <a name="changing-code"></a>Изменение исходного когда

По умолчанию функции не имеют внутри себя реализации - только `pass` в теле функции:

![](../images/starting_guide/initial_function_state.png)

Ваша задача - реализовать функцию по предоставленному описанию лабораторной работы:

![](../images/starting_guide/implemented_function.png)


### <a name="committing-changes"></a>Фиксация изменений с помощью системы контроля версий `git`

Git - система контроля версий, которая позволяет сразу нескольким разработчикам сохранять и отслеживать изменения в файлах проекта.

Сейчас мы зафиксируем изменения, сделанные на предыдущем шаге в файле `main.py`. Чтобы это сделать, выполните следующие шаги:

1. Откройте терминал в среде разработки PyCharm:

    ![](../images/starting_guide/pycharm_open_terminal.png)
2. В терминале выполните команду `git add <путь-до-лабораторной-работы>/main.py`

    ![](../images/starting_guide/git_add.png)
3. В терминале выполните команду `git commit -m "message"`

    ![](../images/starting_guide/git_commit.png)
    * **NB**: В качестве `message` рекомендуется использовать краткое описание тех изменений, которые Вы сделали. Этот текст будет публично доступен!

Больше информации о командах, описанных выше, можно найти в [официальной документации по Git](https://git-scm.com/docs).

### <a name="pushing-changes"></a>Отправка изменений в удалённый форк

После предыдущего шага изменения находятся в состоянии зафиксированных. Они сохранены только у Вас на системе. 
Чтобы отправить их в удалённый (находящийся на платформе GitHub) форк, созданный ранее, выполните следующие шаги:

1. Откройте терминал в среде разработки PyCharm:

    ![](../images/starting_guide/pycharm_open_terminal.png)
2. В терминале выполните команду `git push`

    ![](../images/starting_guide/git_push.png)
3. Откройте главную страницу Вашего форка:

    ![](../images/starting_guide/fork_updated.png)
    * **NB**: Вы увидите сделанный *commit* и сообщение, которое Вы написали.

Больше информации о командах, описанных выше, можно найти в [официальной документации по Git](https://git-scm.com/docs).

## <a name="creating-pull-request"></a>Создание Pull Request

Чтобы менторы смогли увидеть Ваши изменения и провести проверку, Вам нужно создать Pull Request на платформе GitHub.

Чтобы создать Pull Request, выполните следующие шаги:

1. Откройте сайт репозитория, который Вам прислал преподаватель
2. Выберите вкладку Pull Requests

    ![](../images/starting_guide/github_pull_request_highlighted.png)
3. Нажмите кнопку `New pull request`

    ![](../images/starting_guide/github_new_pull_request.png)
4. Нажмите кнопку `compare across forks`

    ![](../images/starting_guide/github_compare_across_forks.png)
5. Нажмите `head repository` и из списка выберите Ваш форк (он будет содержать имя Вашего пользователя)

    ![](../images/starting_guide/github_choose_fork.png)
6. Нажмите кнопку `Create pull request`

    ![](../images/starting_guide/github_create_pull_request_final_step.png)
7. Введите название для Pull Request

    ![](../images/starting_guide/github_name_pull_request.png)
    * **NB**: Имя PR должно соответствовать следующему шаблону: `Laboratory work #X, Name Surname - 2XFPLX`
    * **NB №2**: Пример имени Вы можете увидеть на скриншоте выше.
8. Нажмите `Assignees` и из списка выберите ментора, который указан в таблице успеваемости:

    ![](../images/starting_guide/github_assignees.png)
9. Нажмите кнопку `Create pull request`

    ![](../images/starting_guide/github_create_pull_request_done.png)
    * **NB**: Pull Request появится в списке PR, который находится на странице из шага №2.

## <a name="continue-working"></a>Продолжение работы

Продолжение работы заключается в повторении нескольких шагов:

1. [Вы изменяете исходный код](#changing-code)
2. [Вы фиксируете изменения](#committing-changes)
3. [Вы отправляете изменения в удалённый форк](#pushing-changes)
   * Они автоматически будут обновляться и в Pull Request, который Вы создали
4. Ментор проверяет Ваш код и оставляет комментарии
5. Вы исправляете исходный код согласно комментариям
6. См. шаг №2
