import os
import pandas as pd
import json
import pickle as pkl
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import utils
from tqdm import tqdm
import config
import itertools
import random

from nltk.tokenize.treebank import TreebankWordTokenizer
from preprocessing import clean_text

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data
def dump_pkl(path,info):
    pkl.dump(info,open(path,'wb'))  

class Fake_Data(object):
    def __init__(self,pos_file,length,mode='gen',neg_file=None):
        self.mode=mode
        if mode=='gen':
            self.pos_list=self.read_file(pos_file)
        elif mode=='dis':
            self.pos_list=self.read_file(pos_file)
            self.neg_list=self.read_file(neg_file)
        self.length=length
        self.entries=self.load_entries()
        #print ('The length of entries is:',len(self.entries))
    
    def pad_sent(self,sent):
        if len(sent)>= self.length:
            sent=sent[:self.length]
            sent[-1]=0
        else:
            padding =[0]*(self.length-len(sent))
            sent=sent + padding
        
        target=sent
        sent=[1]+sent
        input_sent=sent[:-1]
        return input_sent,target
    
    def load_entries(self):
        entries=[]
        for tokens in self.pos_list:
            sent,target=self.pad_sent(tokens)
            tokens=torch.from_numpy(np.array(sent,dtype=np.int64)).long()
            target=torch.from_numpy(np.array(target,dtype=np.int64)).long()
            entry={
                'label':1,
                'tokens':tokens,
                'target':target
            }
            entries.append(entry)
        if self.mode=='dis':
            for tokens in self.neg_list:
                sent,target=self.pad_sent(tokens)
                tokens=torch.from_numpy(np.array(sent,dtype=np.int64)).long()
                target=torch.from_numpy(np.array(target,dtype=np.int64)).long()
                entry={
                    'label':0,
                    'tokens':tokens,
                    'target':target
                }
                #print (entry['label'])
                entries.append(entry)
        return entries
        
    def read_file(self,file_path):
        with open(file_path,'r') as f:
            lines=f.readlines()
        lis=[]
        for line in lines:
            l=line.strip().split(' ')
            l=[int(s) for s in l]
            lis.append(l)
        return lis
        
    def __getitem__(self,index):
        entry=self.entries[index]
        target=entry['target']
        tokens=entry['tokens']
        if self.mode=='dis':
            label=torch.from_numpy(np.array(entry['label'],dtype=np.int64)).long()
            #print (entry['label'])
            return target,label
        else:
            return tokens,target
        
    def __len__(self):
        return len(self.entries)
    
class Real_Data(object):
    def __init__(self,pos_file,opt,mode='gen',neg_file=None):
        self.mode=mode
        delete_symbols='@＼・ω+=”“[]^–>\\°<~•≠™ˈʊɒ∞§{}·τα❤☺ɡ|¢→̶`❥━┣┫┗Ｏ►★©―ɪ✔®\x96\x92●£♥➤´¹☕≈÷♡◐║▬′ɔː€۩۞†μ✒➥═☆ˌ◄½ʻπδηλσερνʃ✬ＳＵＰＥＲＩＴ☻±♍µº¾✓◾؟．⬅℅»Вав❣⋅¿¬♫ＣＭβ█▓▒░⇒⭐›¡₂₃❧▰▔◞▀▂▃▄▅▆▇↙γ̄″☹➡«φ⅓„✋：¥̲̅́∙‛◇✏▷❓❗¶˚˙）сиʿ✨。ɑ\x80◕！％¯−ﬂﬁ₁²ʌ¼⁴⁄₄⌠♭✘╪▶☭✭♪☔☠♂☃☎✈✌✰❆☙○‣⚓年∎ℒ▪▙☏⅛ｃａｓǀ℮¸ｗ‚∼‖ℳ❄←☼⋆ʒ⊂、⅔¨͡๏⚾⚽Φ×θ￦？（℃⏩☮⚠月✊❌⭕▸■⇌☐☑⚡☄ǫ╭∩╮，例＞ʕɐ̣Δ₀✞┈╱╲▏▕┃╰▊▋╯┳┊≥☒↑☝ɹ✅☛♩☞ＡＪＢ◔◡↓♀⬆̱ℏ\x91⠀ˤ╚↺⇤∏✾◦♬³の｜／∵∴√Ω¤☜▲↳▫‿⬇✧ｏｖｍ－２０８＇‰≤∕ˆ⚜☁.,?!-;*"…:—()%#$&_/\n🍕\r🐵😑\xa0\ue014\t\uf818\uf04a\xad😢🐶️\uf0e0😜😎👊\u200b\u200e😁عدويهصقأناخلىبمغر😍💖💵Е👎😀😂\u202a\u202c🔥😄🏻💥ᴍʏʀᴇɴᴅᴏᴀᴋʜᴜʟᴛᴄᴘʙғᴊᴡɢ😋👏שלוםבי😱‼\x81エンジ故障\u2009🚌ᴵ͞🌟😊😳😧🙀😐😕\u200f👍😮😃😘אעכח💩💯⛽🚄🏼ஜ😖ᴠ🚲‐😟😈💪🙏🎯🌹😇💔😡\x7f👌ἐὶήιὲκἀίῃἴξ🙄Ｈ😠\ufeff\u2028😉😤⛺🙂\u3000تحكسة👮💙فزط😏🍾🎉😞\u2008🏾😅😭👻😥😔😓🏽🎆🍻🍽🎶🌺🤔😪\x08‑🐰🐇🐱🙆😨🙃💕𝘊𝘦𝘳𝘢𝘵𝘰𝘤𝘺𝘴𝘪𝘧𝘮𝘣💗💚地獄谷улкнПоАН🐾🐕😆ה🔗🚽歌舞伎🙈😴🏿🤗🇺🇸мυтѕ⤵🏆🎃😩\u200a🌠🐟💫💰💎эпрд\x95🖐🙅⛲🍰🤐👆🙌\u2002💛🙁👀🙊🙉\u2004ˢᵒʳʸᴼᴷᴺʷᵗʰᵉᵘ\x13🚬🤓\ue602😵άοόςέὸתמדףנרךצט😒͝🆕👅👥👄🔄🔤👉👤👶👲🔛🎓\uf0b7\uf04c\x9f\x10成都😣⏺😌🤑🌏😯ех😲Ἰᾶὁ💞🚓🔔📚🏀👐\u202d💤🍇\ue613小土豆🏡❔⁉\u202f👠》कर्मा🇹🇼🌸蔡英文🌞🎲レクサス😛外国人关系Сб💋💀🎄💜🤢َِьыгя不是\x9c\x9d🗑\u2005💃📣👿༼つ༽😰ḷЗз▱ц￼🤣卖温哥华议会下降你失去所有的钱加拿大坏税骗子🐝ツ🎅\x85🍺آإشء🎵🌎͟ἔ油别克🤡🤥😬🤧й\u2003🚀🤴ʲшчИОРФДЯМюж😝🖑ὐύύ特殊作戦群щ💨圆明园קℐ🏈😺🌍⏏ệ🍔🐮🍁🍆🍑🌮🌯🤦\u200d𝓒𝓲𝓿𝓵안영하세요ЖљКћ🍀😫🤤ῦ我出生在了可以说普通话汉语好极🎼🕺🍸🥂🗽🎇🎊🆘🤠👩🖒🚪天一家⚲\u2006⚭⚆⬭⬯⏖新✀╌🇫🇷🇩🇪🇮🇬🇧😷🇨🇦ХШ🌐\x1f杀鸡给猴看ʁ𝗪𝗵𝗲𝗻𝘆𝗼𝘂𝗿𝗮𝗹𝗶𝘇𝗯𝘁𝗰𝘀𝘅𝗽𝘄𝗱📺ϖ\u2000үսᴦᎥһͺ\u2007հ\u2001ɩｙｅ൦ｌƽｈ𝐓𝐡𝐞𝐫𝐮𝐝𝐚𝐃𝐜𝐩𝐭𝐢𝐨𝐧Ƅᴨןᑯ໐ΤᏧ௦Іᴑ܁𝐬𝐰𝐲𝐛𝐦𝐯𝐑𝐙𝐣𝐇𝐂𝐘𝟎ԜТᗞ౦〔Ꭻ𝐳𝐔𝐱𝟔𝟓𝐅🐋ﬃ💘💓ё𝘥𝘯𝘶💐🌋🌄🌅𝙬𝙖𝙨𝙤𝙣𝙡𝙮𝙘𝙠𝙚𝙙𝙜𝙧𝙥𝙩𝙪𝙗𝙞𝙝𝙛👺🐷ℋ𝐀𝐥𝐪🚶𝙢Ἱ🤘ͦ💸ج패티Ｗ𝙇ᵻ👂👃ɜ🎫\uf0a7БУі🚢🚂ગુજરાતીῆ🏃𝓬𝓻𝓴𝓮𝓽𝓼☘﴾̯﴿₽\ue807𝑻𝒆𝒍𝒕𝒉𝒓𝒖𝒂𝒏𝒅𝒔𝒎𝒗𝒊👽😙\u200cЛ‒🎾👹⎌🏒⛸公寓养宠物吗🏄🐀🚑🤷操美𝒑𝒚𝒐𝑴🤙🐒欢迎来到阿拉斯ספ𝙫🐈𝒌𝙊𝙭𝙆𝙋𝙍𝘼𝙅ﷻ🦄巨收赢得白鬼愤怒要买额ẽ🚗🐳𝟏𝐟𝟖𝟑𝟕𝒄𝟗𝐠𝙄𝙃👇锟斤拷𝗢𝟳𝟱𝟬⦁マルハニチロ株式社⛷한국어ㄸㅓ니͜ʖ𝘿𝙔₵𝒩ℯ𝒾𝓁𝒶𝓉𝓇𝓊𝓃𝓈𝓅ℴ𝒻𝒽𝓀𝓌𝒸𝓎𝙏ζ𝙟𝘃𝗺𝟮𝟭𝟯𝟲👋🦊多伦🐽🎻🎹⛓🏹🍷🦆为和中友谊祝贺与其想象对法如直接问用自己猜本传教士没积唯认识基督徒曾经让相信耶稣复活死怪他但当们聊些政治题时候战胜因圣把全堂结婚孩恐惧且栗谓这样还♾🎸🤕🤒⛑🎁批判检讨🏝🦁🙋😶쥐스탱트뤼도석유가격인상이경제황을렵게만들지않록잘관리해야합다캐나에서대마초와화약금의품런성분갈때는반드시허된사용🔫👁凸ὰ💲🗯𝙈Ἄ𝒇𝒈𝒘𝒃𝑬𝑶𝕾𝖙𝖗𝖆𝖎𝖌𝖍𝖕𝖊𝖔𝖑𝖉𝖓𝖐𝖜𝖞𝖚𝖇𝕿𝖘𝖄𝖛𝖒𝖋𝖂𝕴𝖟𝖈𝕸👑🚿💡知彼百\uf005𝙀𝒛𝑲𝑳𝑾𝒋𝟒😦𝙒𝘾𝘽🏐𝘩𝘨ὼṑ𝑱𝑹𝑫𝑵𝑪🇰🇵👾ᓇᒧᔭᐃᐧᐦᑳᐨᓃᓂᑲᐸᑭᑎᓀᐣ🐄🎈🔨🐎🤞🐸💟🎰🌝🛳点击查版🍭𝑥𝑦𝑧ＮＧ👣\uf020っ🏉ф💭🎥Ξ🐴👨🤳🦍\x0b🍩𝑯𝒒😗𝟐🏂👳🍗🕉🐲چی𝑮𝗕𝗴🍒ꜥⲣⲏ🐑⏰鉄リ事件ї💊「」\uf203\uf09a\uf222\ue608\uf202\uf099\uf469\ue607\uf410\ue600燻製シ虚偽屁理屈Г𝑩𝑰𝒀𝑺🌤𝗳𝗜𝗙𝗦𝗧🍊ὺἈἡχῖΛ⤏🇳𝒙ψՁմեռայինրւդձ冬至ὀ𝒁🔹🤚🍎𝑷🐂💅𝘬𝘱𝘸𝘷𝘐𝘭𝘓𝘖𝘹𝘲𝘫کΒώ💢ΜΟΝΑΕ🇱♲𝝈↴💒⊘Ȼ🚴🖕🖤🥘📍👈➕🚫🎨🌑🐻𝐎𝐍𝐊𝑭🤖🎎😼🕷ｇｒｎｔｉｄｕｆｂｋ𝟰🇴🇭🇻🇲𝗞𝗭𝗘𝗤👼📉🍟🍦🌈🔭《🐊🐍\uf10aლڡ🐦\U0001f92f\U0001f92a🐡💳ἱ🙇𝗸𝗟𝗠𝗷🥜さようなら🔼'
        self.tokenizer=TreebankWordTokenizer()
        self.remove_dict={ord(c): '' for c in delete_symbols}
        
        if mode=='gen':
            self.pos_list=self.read_file(pos_file)
        elif mode=='dis':
            self.pos_list=self.read_file(pos_file)
            self.neg_list=self.read_file(neg_file)
        self.opt=opt
        self.length=opt.SENT_LEN
        if opt.CREATE_DICT:
            print ('Start to create dictionary...')
            self.create_dictionary()
        else:
            print ('Start loading information of the dictionary...')
            dict_path=os.path.join(opt.DICT_PATH,opt.DATASET+'_dictionary.pkl')
            utils.assert_exits(dict_path)
            created_dict=load_pkl(dict_path)
            self.word2idx=created_dict[0]
            self.idx2word=created_dict[1]
            print ('The length of dictionary is:',len(self.idx2word))
        if self.opt.CREATE_EMB:
            print ('Creating Embedding...')
            self.glove_weights=self.create_embedding()
        self.entries=self.load_entries()
    
    def create_dictionary(self):
        self.word2idx={'<eos>':0,'<start>':1,'UNK':2}
        self.idx2word=['<eos>','<start>','UNK']
        self.dict_token_sent()
    
    def handle_contractions(self,x):
        x = self.tokenizer.tokenize(x)
        return x
    
    def handle_punctuation(self,x):
        x = x.translate(self.remove_dict)
        
        return x
    
    def fix_quote(self,x):
        x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]
        x = ' '.join(x)
        return x
    
    def tokenize(self,sent,add_to_dict):
        #sentence=sent.lower()
        #sentence=sentence.replace(',','').replace('?','').replace('\'s','\'s').replace('.','')
        x = clean_text(sent)
        x = self.handle_punctuation(x)
        x = self.handle_contractions(x)
        sentence = self.fix_quote(x)
        words=sentence.lower().split()
        tokens=[]
        if add_to_dict:
            for w in words:
                if w in self.count_words:
                    self.count_words[w]+=1
                else:
                    self.count_words[w]=1
        else:
            for w in words:
                if w in self.word2idx:
                    tokens.append(self.word2idx[w])
                else:
                    tokens.append(self.word2idx['UNK'])
        return tokens
    
    def dict_token_sent(self):
        dataset=json.load(open(self.opt.REAL_FILE,'r'))
        self.count_words={}
        for tweet in dataset:
            self.tokenize(tweet,True) 
        start_num=3
        final=sorted(self.count_words.items(), key=lambda d:d[1], reverse = True)[:self.opt.VOC_SIZE-3]
        for word in final:
            #print (word[0])
            self.word2idx[word[0]]=start_num
            self.idx2word.append(word[0])
            start_num+=1
        print ('Dictionary Creation done!')
        print ('The lenght of the dictionary is:',len(self.idx2word))
        dump_pkl(os.path.join(self.opt.DICT_PATH,self.opt.DATASET+'_dictionary.pkl'),[self.word2idx,self.idx2word])
    
    def create_embedding(self):
        word2emb={}
        with open(self.opt.GLOVE_PATH,'r') as f:
            entries=f.readlines()
        emb_dim=len(entries[0].split(' '))-1
        weights=np.zeros((len(self.idx2word),emb_dim),dtype=np.float32)
        for entry in entries:
            word=entry.split(' ')[0]
            word2emb[word]=np.array(list(map(float,entry.split(' ')[1:])))
        for idx,word in enumerate(self.idx2word):
            if word not in word2emb:
                continue
            weights[idx]=word2emb[word]
        emb_path=os.path.join(self.opt.DICT_PATH,self.opt.DATASET+'_glove_embedding.npy')
        np.save(emb_path,weights)
        return weights
    
    def read_file(self,file_path):
        entity=json.load(open(file_path,'r'))
        return entity
    
    def pad_sent(self,sent):
        if len(sent)>= self.length:
            sent=sent[:self.length]
            sent[-1]=0
        else:
            padding =[0]*(self.length-len(sent))
            sent=sent + padding
        target=sent
        sent=[1]+sent
        input_sent=sent[:-1]
        return input_sent,target
    
    def load_entries(self):
        entries=[]
        for sent in self.pos_list:
            sent=self.tokenize(sent,False)
            token,target=self.pad_sent(sent)
            token=np.array(token,dtype=np.int64)
            target=np.array(target,dtype=np.int64)
            entry={
                'label':1,
                'sent':sent,
                'tokens':token,
                'target':target
            }
            entries.append(entry)
        if self.mode=='dis':
            for sent in self.neg_list:
                sent=self.tokenize(sent,False)
                token,target=self.pad_sent(sent)
                token=np.array(token,dtype=np.int64)
                target=np.array(target,dtype=np.int64)
                entry={
                    'label':0,
                    'sent':sent,
                    'tokens':token,
                    'target':target
                }
                entries.append(entry)
        return entries
    
    def __getitem__(self,index):
        entry=self.entries[index]
        target=entry['target']
        tokens=entry['tokens']
        if self.mode=='dis':
            label=torch.from_numpy(np.array(entry['label'],dtype=np.int64)).long()
            #print (entry['label'])
            return target,label
        else:
            return tokens,target
        
        
    def __len__(self):
        return len(self.entries)