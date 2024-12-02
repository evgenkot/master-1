from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///items.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Float, nullable=False)
    availability = db.Column(db.Boolean, default=True)


with app.app_context():
    db.create_all()


@app.route('/')
def index():
    items = Item.query.all()
    return render_template('index.html', items=items)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/edit/<int:item_id>', methods=['GET', 'POST'])
@app.route('/edit', methods=['GET', 'POST'])
def edit(item_id=None):
    if request.method == 'POST':
        name = request.form['name']
        price = float(request.form['price'])
        availability = 'availability' in request.form

        if item_id is not None:
            item = Item.query.get(item_id)
            item.name = name
            item.price = price
            item.availability = availability
        else:
            item = Item(name=name, price=price, availability=availability)
            db.session.add(item)

        db.session.commit()
        return redirect(url_for('index'))

    item = Item.query.get(item_id) if item_id else None
    items = Item.query.all()
    return render_template('edit.html', item=item, items=items)


@app.route('/delete/<int:item_id>', methods=['POST'])
def delete_item(item_id):
    item = Item.query.get(item_id)
    if item:
        db.session.delete(item)
        db.session.commit()
    return redirect(url_for('index'))

@app.route('/order/<int:item_id>', methods=['POST'])
def order(item_id):
    item = Item.query.get(item_id)
    if item and item.availability:
        return render_template('order_confirmation.html', item=item)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

