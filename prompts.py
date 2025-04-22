# Top 20 color words in English
color_words = ['red', 'green', 'blue', 
               'yellow', 'pink', 'black', 
               'white', 'gray', 'purple', 
               'orange', 'brown', 'beige', 
               'gold', 'silver', 'navy', 
               'teal', 'olive', 'maroon', 
               'violet', 'cyan']
# Top 20 animal words in English
animal_words = ['dog', 'cat', 'bird', 
                'fish', 'horse', 'sheep', 
                'cow', 'chicken', 'pig', 
                'duck', 'goat', 'rabbit', 
                'deer', 'bear', 'elephant', 
                'tiger', 'lion', 'monkey', 
                'wolf', 'fox']

# Prefaces to be used for prompting: will appear before train samples
preface1 = "<|endoftext|> "
preface2 = ("<|endoftext|> There are 5 kinds of animals that come in 5 " 
            "different colors (25 total colored animals). "
            "Here are the locations (in xy-coordinates) " 
            "of some of the colored animals: ")
preface3 = ("There are 5 kinds of animals that come in 5 " 
            "different colors (25 total colored animals). "
            "Here are the locations (in xy-coordinates) " 
            "of some of the colored animals: ")
preface4 = ""
preface5 = ("<s>[INST] <<SYS>>\nYou are a clever problem solver. You can see "
            "past the surface-level details of a problem to find the correct "
            "solution. Your reasoning abilities will be tested. Keep your "
            "answers short and precise.\n<</SYS>>\n\nThere are 5 kinds of "
            "animals that come in 5 different colors (25 total colored "
            "animals). Here are the locations (in xy-coordinates) " 
            "of some of the colored animals: ")
preface6 = ("<s>[INST] <<SYS>>\n\n<</SYS>>\n\nThere are 5 kinds of "
            "animals that come in 5 different colors (25 total colored "
            "animals). Here are the locations (in xy-coordinates) " 
            "of some of the colored animals: ")
preface7 = ("<s>[INST] ")
preface8 = ("<s>[INST] <<SYS>>\n\n<</SYS>>\n\n ")
preface9 = ("<s>[INST] <<SYS>>\nThere are 5 kinds of "
            "animals that come in 5 different colors (25 total colored "
            "animals). Here are the locations (in xy-coordinates) " 
            "of some of the colored animals: \n<</SYS>>\n\n ")
prefaces = [preface1, preface2, preface3, preface4, 
            preface5, preface6, preface7, preface8, preface9]

# Sample templates for grid task
template1 = "<color> <animal> : <x> <y>"
template2 = "<color> , <animal> : <x> <y>"
template3 = "( <color> <animal> ) : <x> <y>"
template4 = "( <color> , <animal> ) : <x> <y>"
templates = [template1, template2, template3, template4]

# Strings for separating train samples
sep1 = " <sep> "
sep2 = " ; "
sep3 = " , "
sep4 = " | "
seps = [sep1, sep2, sep3, sep4]

# Test prefaces
test_preface1 = ""
test_preface2 = "What is the location of the following animal? "
test_preface3 = "What is the location of the following colored animal? "
test_preface4 = "Give me the location of one of the colored animals. [/INST] "
test_preface5 = "[/INST] "
test_prefaces= [test_preface1, test_preface2, test_preface3, 
                test_preface4, test_preface5]