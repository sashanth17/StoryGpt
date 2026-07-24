import json


class Tokenizer:


    def __init__(
        self,
        vocab_file="vocab.json",
        merges_file="merges.txt"
    ):


        # token -> id

        self.token_to_id = json.load(
            open(
                vocab_file,
                encoding="utf-8"
            )
        )


        # id -> token

        self.id_to_token = {
            v:k
            for k,v in self.token_to_id.items()
        }



        # merge priority

        self.merges = {}


        with open(
            merges_file,
            encoding="utf-8"
        ) as f:


            for index,line in enumerate(f):

                a,b=line.strip().split()

                self.merges[(a,b)] = index



    def get_pairs(self,tokens):

        pairs=set()

        for i in range(len(tokens)-1):

            pairs.add(
                (
                    tokens[i],
                    tokens[i+1]
                )
            )

        return pairs



    def encode(self,text):


        # start with characters

        tokens=list(text)



        while True:


            pairs=self.get_pairs(tokens)



            # find available merges

            candidates=[
                pair
                for pair in pairs
                if pair in self.merges
            ]



            if not candidates:
                break



            # choose earliest learned merge

            best_pair=min(
                candidates,
                key=lambda x:self.merges[x]
            )



            new_tokens=[]

            i=0


            while i<len(tokens):


                if (
                    i<len(tokens)-1
                    and tokens[i]==best_pair[0]
                    and tokens[i+1]==best_pair[1]
                ):

                    new_tokens.append(
                        best_pair[0]+best_pair[1]
                    )

                    i+=2


                else:

                    new_tokens.append(
                        tokens[i]
                    )

                    i+=1



            tokens=new_tokens



        return [
            self.token_to_id[token]
            for token in tokens
        ]




    def decode(self,ids):

        tokens=[
            self.id_to_token[i]
            for i in ids
        ]

        return "".join(tokens)