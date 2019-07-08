from flair.data import Sentence
from flair.models import SequenceTagger

# load the model you trained
model = SequenceTagger.load('C:\Projects\SAKI_NLP\models/flair_best-model_33.pt')

sent = "Afreen Jamadar\nActive member of IIIT Committee in Third year\n\nSangli, Maharashtra - Email me on Indeed: indeed.com/r/Afreen-Jamadar/8baf379b705e37c6\n\nI wish to use my knowledge, skills and conceptual understanding to create excellent team\nenvironments and work consistently achieving organization objectives believes in taking initiative\nand work to excellence in my work.\n\nWORK EXPERIENCE\n\nActive member of IIIT Committee in Third year\n\nCisco Networking -  Kanpur, Uttar Pradesh\n\norganized by Techkriti IIT Kanpur and Azure Skynet.\nPERSONALLITY TRAITS:\n\u2022 Quick learning ability\n\u2022 hard working\n\nEDUCATION\n\nPG-DAC\n\nCDAC ACTS\n\n2017\n\nBachelor of Engg in Information Technology\n\nShivaji University Kolhapur -  Kolhapur, Maharashtra\n\n2016\n\nSKILLS\n\nDatabase (Less than 1 year), HTML (Less than 1 year), Linux. (Less than 1 year), MICROSOFT\nACCESS (Less than 1 year), MICROSOFT WINDOWS (Less than 1 year)\n\nADDITIONAL INFORMATION\n\nTECHNICAL SKILLS:\n\n\u2022 Programming Languages: C, C++, Java, .net, php.\n\u2022 Web Designing: HTML, XML\n\u2022 Operating Systems: Windows [\u2026] Windows Server 2003, Linux.\n\u2022 Database: MS Access, MS SQL Server 2008, Oracle 10g, MySql.\n\n"


# create example sentence
sentence = Sentence(sent)

# predict tags and print
model.predict(sentence)

print(sentence.to_tagged_string())
from flair.datasets import WIKINER_ENGLISH
x = WIKINER_ENGLISH()