# Template-based NLG on Multiwoz

Template NLG for Multiwoz dataset. The templates are extracted from data and modified manually.

## How to run

There are three mode:

- `auto`: templates extracted from data without manual modification, may have no match (return 'None');
- `manual`: templates with manual modification, sometimes verbose;
- `auto_manual`: use auto templates first. When fails, use manual templates.

Example:

```python
from tatk.nlg.template_nlg.multiwoz.nlg import TemplateNLG

# dialog act
dialog_acts = {'Train-Inform': [['Day', 'wednesday'], ['Leave', '10:15']]}
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
{'Train-Inform': [['Day', 'wednesday'], ['Leave', '10:15']]}
manual      :  The train is for wednesday you are all set. There is a train meeting your criteria and is leaving at 10:15 .
auto        :  I can help you with that . one leaves wednesday at 10:15 , is that time okay for you ?
auto_manual :  There is a train on wednesday at 10:15 .
```

## Templates

We select the utterances that have only one dialog act to extract templates. For `auto` mode, the templates may have several slot, while for `manual` mode, the templates only have one slot. As a result, `auto` templates can fail when some slot combination don't appear in dataset, while for `manual` mode, we generate utterance slot by slot, which could not fail but may be verbose. Notice that `auto` templates could be inappropriate.
