# Template-based NLG on Camrest

Template NLG for Camrest dataset. The templates are extracted from data and modified manually. In addition, we have added some templates from the _Restaurant_ domain in the [Multiwoz templates](../multiwoz/README.md).

## How to run

There are three mode:

- `auto`: templates extracted from data without manual modification, may have no match (return 'None');
- `manual`: templates with manual modification, sometimes verbose;
- `auto_manual`: use auto templates first. When fails, use manual templates.

Example:

```python
from tatk.nlg.template_nlg.camrest.nlg import TemplateNLG

# dialog act
dialog_acts = {'inform': [['pricerange', 'cheap'], ['area', 'west']]}
print(dialog_acts)

# system model for manual, auto, auto_manual
nlg_sys_manual = TemplateNLG(is_user=False, mode='manual')
nlg_sys_auto = TemplateNLG(is_user=False, mode='auto')
nlg_sys_auto_manual = TemplateNLG(is_user=False, mode='auto_manual')

# generate
print('manual      : ', nlg_sys_manual.generate(dialog_acts))
print('auto        : ', nlg_sys_auto.generate(dialog_acts))
print('auto_manual : ', nlg_sys_auto_manual.generate(dialog_acts))
```
Result:
```
{'inform': [['pricerange', 'cheap'], ['area', 'west']]}
manual      :  They are in the cheap price range . It is located in the west part of town.
auto        :  Both are in the west and in the cheap price range .
auto_manual :  Yes , it is cheap and in the west .
```

## Templates

The camrest template set consists of two sources.
1. Extract data from the Camrest dataset. The extraction method is the same as the [Multiwoz](../multiwoz/README.md).
2. Added templates from the _Restaurant_ domain in the [Multiwoz templates](../multiwoz/README.md).
