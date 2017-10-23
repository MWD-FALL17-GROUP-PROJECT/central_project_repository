# central_project_repository
This is a README file for our Phase 2 MWD Project.

The code should run in python 3 and above.
We have written and tested it for versions 3.5 to 3.6.1.

Dependencies of the package are as follows:
1. scikit-learn
2. Gensim
3. Pandas
4. sk-tensor: Instructions to install:
	      1. Download the zip file scikit-tensor-master.zip present in the additional_files folder
              2. Unzip in shell and go inside the unzipped folder via the shell.
              3. Run python setup.py install
5. numpy
6. Anaconda (will cover pandas, numpy and scikit-learn)

Our project runs on imdb data which should be present in C:\MWD Project\phase1_dataset

This project is run using a command line interface using interface.py which works in the following manner:
    For Tasks 1,2 and 3:
    interface.py <TASK_NUMBER> <SUBTASK> <ARGUMENTS>
        where TASK_NUMBER can be task1,task2 or task3
	SUBTASK can be a,b,c or d
	ARGUMENTS can be
            For task 1, it would be OBJECT METHOD where:
                OBJECT could be 
                    For task 1a &1b: Any of the available genres of movies in the corpus 
                    For task 1c: Any of the available actor_id in the corpus
                    For task 1d: Any of the available movie_id in the corpus
            For task2, it would be:
                    for task 2a & 2b, neither would be required 
                    for task 2c & 2d, it would be SPACE
                    For 2c SPACE can be: actor, movie or year
                    For 2d SPACE can be: tag, movie, rating
            For task 3,both a & b,  it would be:
                    A list of actor_id comma separated e,g a1,a2,a3 with no space in between
            For Tasks 4, it would be:
                    interface.py <TASK_NUMBER> <USER_ID> : USER_ID would be the user_id  of the user for whom we’re recommending movies 
                    

Note: LDA takes 2-4 minutes to run for task1.
